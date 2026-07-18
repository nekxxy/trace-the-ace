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
from trace_ace.provenance import (
    SEMANTIC_CV_SOURCE_FILES,
    SPARSE_CV_SOURCE_FILES,
    VALIDATION_PIPELINE_SOURCE_FILES,
    file_sha256,
    response_identity_sha256,
    trace_ace_source_sha256,
    validate_store_manifest_source,
)
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
    args = parser.parse_args()
    sparse_oof_path = args.run_dir / "sparse_oof.parquet"
    semantic_oof_path = args.run_dir / "semantic_oof.parquet"
    sparse_metrics_path = args.run_dir / "sparse_metrics.json"
    semantic_metrics_path = args.run_dir / "semantic_metrics.json"
    sparse_manifest_path = args.sparse_store / "manifest.json"
    semantic_manifest_path = args.semantic_store / "manifest.json"
    sparse = pd.read_parquet(sparse_oof_path).sort_values("row_id")
    semantic = pd.read_parquet(semantic_oof_path).sort_values("row_id")
    if not np.array_equal(sparse["row_id"], semantic["row_id"]):
        raise ValueError("OOF row IDs differ")
    if not np.array_equal(sparse["is_correct"], semantic["is_correct"]):
        raise ValueError("OOF targets differ")
    config = ModelConfig()
    responses = pq.read_table(args.responses).to_pandas().sort_values("row_id")
    response_digest = response_identity_sha256(responses)
    sparse_manifest = json.loads(sparse_manifest_path.read_text(encoding="utf-8"))
    semantic_manifest = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
    validate_store_manifest_source(sparse_manifest, store="sparse")
    validate_store_manifest_source(semantic_manifest, store="semantic")
    for name, manifest in (("sparse", sparse_manifest), ("semantic", semantic_manifest)):
        source = manifest.get("source")
        if not isinstance(source, dict) or source.get("config") != config.to_dict():
            raise ValueError(f"{name} store configuration differs from ensemble evaluation")
        if source.get("response_identity_sha256") != response_digest:
            raise ValueError(f"responses and {name} store identities differ")
    response_view_hash = sparse_manifest["source"]["source_sha256"]
    if semantic_manifest["source"]["source_sha256"] != response_view_hash:
        raise ValueError("sparse and semantic stores derive from different response views")
    sparse_metrics = json.loads(sparse_metrics_path.read_text(encoding="utf-8"))
    semantic_metrics = json.loads(semantic_metrics_path.read_text(encoding="utf-8"))
    _validate_component_evidence(
        evidence=sparse_metrics,
        name="sparse",
        expected_config=config.to_dict(),
        response_digest=response_digest,
        response_view_hash=response_view_hash,
        store_manifest_path=sparse_manifest_path,
        oof_path=sparse_oof_path,
        source_files=SPARSE_CV_SOURCE_FILES,
        runner_path=PROJECT_ROOT / "scripts/run_sparse_cv.py",
    )
    _validate_component_evidence(
        evidence=semantic_metrics,
        name="semantic",
        expected_config=config.to_dict(),
        response_digest=response_digest,
        response_view_hash=response_view_hash,
        store_manifest_path=semantic_manifest_path,
        oof_path=semantic_oof_path,
        source_files=SEMANTIC_CV_SOURCE_FILES,
        runner_path=PROJECT_ROOT / "scripts/run_semantic_cv.py",
    )
    probability = blend_probabilities(
        sparse["full_probability"].to_numpy(),
        sparse["role_probability"].to_numpy(),
        semantic["semantic_probability"].to_numpy(),
        weights=(config.full_weight, config.role_weight, config.semantic_weight),
        probability_floor=config.probability_floor,
    )
    targets = sparse["is_correct"].to_numpy()
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
        "response_identity_sha256": response_digest,
        "response_view_sha256": response_view_hash,
        "sparse_store_manifest_sha256": file_sha256(sparse_manifest_path),
        "semantic_store_manifest_sha256": file_sha256(semantic_manifest_path),
        "sparse_metrics_sha256": file_sha256(sparse_metrics_path),
        "semantic_metrics_sha256": file_sha256(semantic_metrics_path),
        "sparse_oof_sha256": file_sha256(sparse_oof_path),
        "semantic_oof_sha256": file_sha256(semantic_oof_path),
        "validation_source_sha256": trace_ace_source_sha256(
            VALIDATION_PIPELINE_SOURCE_FILES
        ),
        "evaluator_source_sha256": file_sha256(Path(__file__)),
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


def _validate_component_evidence(
    *,
    evidence: object,
    name: str,
    expected_config: dict[str, object],
    response_digest: str,
    response_view_hash: str,
    store_manifest_path: Path,
    oof_path: Path,
    source_files: tuple[str, ...],
    runner_path: Path,
) -> None:
    """Require a component OOF file to match its exact training provenance."""

    if not isinstance(evidence, dict):
        raise ValueError(f"{name} validation evidence is not an object")
    expected = {
        "config": expected_config,
        "response_identity_sha256": response_digest,
        "response_view_sha256": response_view_hash,
        "store_manifest_sha256": file_sha256(store_manifest_path),
        "training_source_sha256": trace_ace_source_sha256(source_files),
        "runner_source_sha256": file_sha256(runner_path),
        "oof_sha256": file_sha256(oof_path),
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ValueError(f"{name} validation {field} differs from current inputs")


if __name__ == "__main__":
    raise SystemExit(main())
