#!/usr/bin/env python3
"""Encode response contexts with the pinned local BGE model."""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_ace.semantic_store import build_semantic_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data/interim/response_views/response_views.parquet",
    )
    parser.add_argument(
        "--asset",
        type=Path,
        default=PROJECT_ROOT / "assets/bge-base-en-v1.5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/semantic_store",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--read-batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_semantic_store(
        args.source,
        args.asset,
        args.output_dir,
        device=args.device,
        encode_batch_size=args.encode_batch_size,
        read_batch_size=args.read_batch_size,
        force=args.force,
        progress=lambda message: print(message, flush=True),
    )
    print(
        f"semantic store: {result.status}; responses={result.response_count}; "
        f"objectives={result.objective_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
