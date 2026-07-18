#!/usr/bin/env python3
"""Run objective-disjoint, validation-session-purged sparse CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trace_ace.config import ModelConfig
from trace_ace.provenance import (
    SPARSE_CV_SOURCE_FILES,
    file_sha256,
    response_identity_sha256,
    trace_ace_source_sha256,
)
from trace_ace.sparse_store import load_dense_by_row
from trace_ace.training import train_sparse_cv
from trace_ace.validation import evaluate_probabilities, objective_disjoint_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        default=PROJECT_ROOT / "data/interim/cache/responses.parquet",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sparse_store",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/cleanroom_v02",
    )
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    responses = pq.read_table(args.responses).to_pandas().sort_values("row_id")
    if not np.array_equal(responses["row_id"].to_numpy(), np.arange(len(responses))):
        raise ValueError("responses row_id must be contiguous")
    store_manifest_path = args.store_dir / "manifest.json"
    store_manifest = json.loads(store_manifest_path.read_text())
    response_digest = response_identity_sha256(responses)
    if store_manifest["source"]["response_identity_sha256"] != response_digest:
        raise ValueError("responses and sparse store identities differ")
    config = ModelConfig(sparse_epochs=args.epochs)
    folds = objective_disjoint_folds(responses)
    result = train_sparse_cv(
        store_dir=args.store_dir,
        folds=folds,
        config=config,
        checkpoint_dir=args.run_dir / "checkpoints",
        progress=lambda message: print(message, flush=True),
    )
    targets, _ = load_dense_by_row(args.store_dir)
    if not np.array_equal(targets, responses["is_correct"].to_numpy(dtype=np.int8)):
        raise ValueError("responses and sparse store targets differ")
    metrics = {
        "config": config.to_dict(),
        "protocol": "objective-disjoint SGKF with validation-session purge",
        "response_identity_sha256": response_digest,
        "response_view_sha256": store_manifest["source"]["source_sha256"],
        "store_manifest_sha256": file_sha256(store_manifest_path),
        "training_source_sha256": trace_ace_source_sha256(
            SPARSE_CV_SOURCE_FILES
        ),
        "runner_source_sha256": file_sha256(Path(__file__)),
        "full_oof": evaluate_probabilities(targets, result.full_oof),
        "role_oof": evaluate_probabilities(targets, result.role_oof),
        "folds": list(result.fold_metrics),
    }
    oof_path = args.run_dir / "sparse_oof.parquet"
    pd.DataFrame(
        {
            "row_id": np.arange(len(targets), dtype=np.int64),
            "is_correct": targets,
            "full_probability": result.full_oof,
            "role_probability": result.role_oof,
        }
    ).to_parquet(oof_path, index=False)
    metrics["oof_sha256"] = file_sha256(oof_path)
    (args.run_dir / "sparse_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"full_oof": metrics["full_oof"], "role_oof": metrics["role_oof"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
