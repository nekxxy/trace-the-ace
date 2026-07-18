#!/usr/bin/env python3
"""Run repeated semantic-family-disjoint ensemble stress tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trace_ace.config import ModelConfig, SEMANTIC_PROTOCOLS
from trace_ace.ensemble import blend_probabilities
from trace_ace.provenance import (
    VALIDATION_PIPELINE_SOURCE_FILES,
    file_sha256,
    response_identity_sha256,
    trace_ace_source_sha256,
    validate_store_manifest_source,
)
from trace_ace.semantic_store import load_semantic_arrays
from trace_ace.semantic_training import train_semantic_cv
from trace_ace.sparse_store import load_dense_by_row
from trace_ace.training import train_sparse_cv
from trace_ace.validation import (
    assign_semantic_families,
    evaluate_probabilities,
    semantic_family_disjoint_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        default=PROJECT_ROOT / "data/interim/cache/responses.parquet",
    )
    parser.add_argument(
        "--sparse-store",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sparse_store",
    )
    parser.add_argument(
        "--semantic-store",
        type=Path,
        default=PROJECT_ROOT / "data/processed/semantic_store",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/robust_cleanroom_v02",
    )
    parser.add_argument("--protocol", choices=[name for name, _, _ in SEMANTIC_PROTOCOLS])
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    responses = pq.read_table(args.responses).to_pandas().sort_values("row_id")
    if not np.array_equal(responses["row_id"], np.arange(len(responses))):
        raise ValueError("responses row_id must be contiguous")
    semantic_arrays = load_semantic_arrays(args.semantic_store)
    targets, _ = load_dense_by_row(args.sparse_store)
    response_digest = response_identity_sha256(responses)
    sparse_manifest_path = args.sparse_store / "manifest.json"
    semantic_manifest_path = args.semantic_store / "manifest.json"
    sparse_manifest = json.loads(sparse_manifest_path.read_text())
    semantic_manifest = json.loads(semantic_manifest_path.read_text())
    validate_store_manifest_source(sparse_manifest, store="sparse")
    validate_store_manifest_source(semantic_manifest, store="semantic")
    if {
        sparse_manifest["source"]["response_identity_sha256"],
        semantic_manifest["source"]["response_identity_sha256"],
    } != {response_digest}:
        raise ValueError("responses, sparse store, and semantic store identities differ")
    expected_targets = responses["is_correct"].to_numpy(dtype=np.int8)
    if not np.array_equal(targets, expected_targets) or not np.array_equal(
        semantic_arrays.targets, expected_targets
    ):
        raise ValueError("responses and feature-store targets differ")
    config = ModelConfig(sparse_epochs=args.epochs)
    if sparse_manifest["source"]["config"] != config.to_dict():
        raise ValueError("sparse store configuration differs from robust validation")
    if semantic_manifest["source"]["config"] != config.to_dict():
        raise ValueError("semantic store configuration differs from robust validation")
    response_view_hash = sparse_manifest["source"]["source_sha256"]
    if semantic_manifest["source"]["source_sha256"] != response_view_hash:
        raise ValueError("sparse and semantic stores derive from different response views")
    sparse_manifest_hash = file_sha256(sparse_manifest_path)
    semantic_manifest_hash = file_sha256(semantic_manifest_path)
    validation_source_hash = trace_ace_source_sha256(
        VALIDATION_PIPELINE_SOURCE_FILES
    )
    runner_source_hash = file_sha256(Path(__file__))
    available = {name: (families, seed) for name, families, seed in SEMANTIC_PROTOCOLS}
    selected = [args.protocol] if args.protocol else list(available)
    summary: dict[str, object] = {
        "config": config.to_dict(),
        "response_identity_sha256": response_digest,
        "response_view_sha256": response_view_hash,
        "sparse_store_manifest_sha256": sparse_manifest_hash,
        "semantic_store_manifest_sha256": semantic_manifest_hash,
        "validation_source_sha256": validation_source_hash,
        "runner_source_sha256": runner_source_hash,
        "protocols": {},
    }

    for protocol_name in selected:
        family_count, family_seed = available[protocol_name]
        families = assign_semantic_families(
            semantic_arrays.objective_embeddings,
            objective_ids=semantic_arrays.objective_ids,
            n_families=family_count,
            seed=family_seed,
        )
        folds = semantic_family_disjoint_folds(
            responses,
            families,
            seed=config.seed + family_seed,
            protocol_name=protocol_name,
        )
        protocol_dir = args.run_dir / protocol_name
        protocol_dir.mkdir(parents=True, exist_ok=True)
        sparse_result = train_sparse_cv(
            store_dir=args.sparse_store,
            folds=folds,
            config=config,
            checkpoint_dir=protocol_dir / "checkpoints",
            progress=lambda message, name=protocol_name: print(f"{name}: {message}", flush=True),
        )
        semantic_result = train_semantic_cv(
            store_dir=args.semantic_store,
            folds=folds,
            config=config,
            checkpoint_dir=protocol_dir / "checkpoints",
            progress=lambda message, name=protocol_name: print(f"{name}: {message}", flush=True),
        )
        ensemble = blend_probabilities(
            sparse_result.full_oof,
            sparse_result.role_oof,
            semantic_result.oof,
            weights=(config.full_weight, config.role_weight, config.semantic_weight),
            probability_floor=config.probability_floor,
        )
        baseline = np.full(len(targets), np.nan, dtype=np.float64)
        for fold in folds:
            baseline[fold.validation_mask] = float(targets[fold.train_mask].mean())
        if not np.isfinite(baseline).all():
            raise ValueError(f"{protocol_name}: fold baseline did not cover every row")
        metrics = {
            "protocol": protocol_name,
            "response_identity_sha256": response_digest,
            "response_view_sha256": response_view_hash,
            "sparse_store_manifest_sha256": sparse_manifest_hash,
            "semantic_store_manifest_sha256": semantic_manifest_hash,
            "validation_source_sha256": validation_source_hash,
            "runner_source_sha256": runner_source_hash,
            "baseline": evaluate_probabilities(targets, baseline),
            "full": evaluate_probabilities(targets, sparse_result.full_oof),
            "role": evaluate_probabilities(targets, sparse_result.role_oof),
            "semantic": evaluate_probabilities(targets, semantic_result.oof),
            "ensemble": evaluate_probabilities(targets, ensemble),
        }
        component_predictions = {
            "baseline": baseline,
            "full": sparse_result.full_oof,
            "role": sparse_result.role_oof,
            "semantic": semantic_result.oof,
            "ensemble": ensemble,
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
        metrics["folds"] = fold_metrics
        metrics["fold_dispersion"] = {
            component: {
                **{
                    f"{metric}_std": float(
                        np.std([record[component][metric] for record in fold_metrics])
                    )
                    for metric in metric_names
                },
                "worst_fold_log_loss": float(
                    max(record[component]["log_loss"] for record in fold_metrics)
                ),
            }
            for component in component_predictions
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
        oof_path = protocol_dir / "oof.parquet"
        pd.DataFrame(
            {
                "row_id": np.arange(len(targets), dtype=np.int64),
                "is_correct": targets,
                "baseline_probability": baseline,
                "full_probability": sparse_result.full_oof,
                "role_probability": sparse_result.role_oof,
                "semantic_probability": semantic_result.oof,
                "ensemble_probability": ensemble,
            }
        ).to_parquet(oof_path, index=False)
        metrics["oof_sha256"] = file_sha256(oof_path)
        summary["protocols"][protocol_name] = metrics
        (protocol_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({protocol_name: metrics["ensemble"]}), flush=True)

    protocol_metrics = summary["protocols"]
    summary["promotion_gate"] = {
        "criterion": (
            "in every repeated semantic-family protocol, ensemble log loss is "
            "strictly lower than the fold-prior baseline and every component"
        ),
        "passed": bool(
            protocol_metrics
            and all(record["promotion_gate"]["passed"] for record in protocol_metrics.values())
        ),
        "worst_protocol": max(
            protocol_metrics,
            key=lambda name: protocol_metrics[name]["ensemble"]["log_loss"],
        ),
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
