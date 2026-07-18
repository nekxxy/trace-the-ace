#!/usr/bin/env python3
"""Fit the memory-bounded sparse baseline after feature-store construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from trace_ace.config import ModelConfig
from trace_ace.training import train_final_sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sparse_store",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models/sparse_baseline.joblib",
    )
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    if not (args.store_dir / "manifest.json").is_file():
        raise FileNotFoundError("build the sparse store before training the baseline")
    model = train_final_sparse(
        store_dir=args.store_dir,
        config=ModelConfig(sparse_epochs=args.epochs),
        progress=lambda message: print(message, flush=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".joblib.tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(args.output)
    print(f"sparse baseline: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
