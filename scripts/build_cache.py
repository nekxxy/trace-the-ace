#!/usr/bin/env python3
"""Build the deterministic Trace the Ace training cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trace_ace.cache import build_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build memory-bounded response and session Parquet caches."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing train_features*, train_labels*, and transcripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "cache",
        help="Destination for Parquet caches and the cache manifest.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild an unchanged cache.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Maximum number of aggregated session rows buffered before each write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_cache(
        args.raw_dir,
        args.output_dir,
        force=args.force,
        batch_size=args.batch_size,
        progress=lambda message: print(message, flush=True),
    )
    print(
        f"cache: {result.status}; responses={result.response_count}; "
        f"sessions={result.session_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
