#!/usr/bin/env python3
"""Build disk-backed sparse matrices once for memory-bounded CV."""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_ace.sparse_store import build_sparse_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data/interim/response_views/response_views.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sparse_store",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_sparse_store(
        args.source,
        args.output_dir,
        batch_size=args.batch_size,
        force=args.force,
        progress=lambda message: print(message, flush=True),
    )
    print(
        f"sparse store: {result.status}; parts={result.part_count}; "
        f"responses={result.response_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
