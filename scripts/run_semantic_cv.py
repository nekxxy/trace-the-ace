#!/usr/bin/env python3
"""Run objective-disjoint BGE semantic interaction CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trace_ace.config import ModelConfig
from trace_ace.provenance import response_identity_sha256
from trace_ace.semantic_store import load_semantic_arrays
from trace_ace.semantic_training import train_semantic_cv
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
        default=PROJECT_ROOT / "data/processed/semantic_store",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/cleanroom_v02",
    )
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    responses = pq.read_table(args.responses).to_pandas().sort_values("row_id")
    if not np.array_equal(responses["row_id"].to_numpy(), np.arange(len(responses))):
        raise ValueError("responses row_id must be contiguous")
    store_manifest = json.loads((args.store_dir / "manifest.json").read_text())
    response_digest = response_identity_sha256(responses)
    if store_manifest["source"]["response_identity_sha256"] != response_digest:
        raise ValueError("responses and semantic store identities differ")
    folds = objective_disjoint_folds(responses)
    config = ModelConfig()
    result = train_semantic_cv(
        store_dir=args.store_dir,
        folds=folds,
        config=config,
        checkpoint_dir=args.run_dir / "checkpoints",
        progress=lambda message: print(message, flush=True),
    )
    arrays = load_semantic_arrays(args.store_dir)
    if not np.array_equal(arrays.targets, responses["is_correct"].to_numpy(dtype=np.int8)):
        raise ValueError("responses and semantic store targets differ")
    metrics = {
        "protocol": "objective-disjoint SGKF with validation-session purge",
        "config": config.to_dict(),
        "response_identity_sha256": response_digest,
        "semantic_oof": evaluate_probabilities(arrays.targets, result.oof),
        "folds": list(result.fold_metrics),
    }
    pd.DataFrame(
        {
            "row_id": np.arange(len(arrays.targets), dtype=np.int64),
            "is_correct": arrays.targets,
            "semantic_probability": result.oof,
        }
    ).to_parquet(args.run_dir / "semantic_oof.parquet", index=False)
    (args.run_dir / "semantic_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"semantic_oof": metrics["semantic_oof"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
