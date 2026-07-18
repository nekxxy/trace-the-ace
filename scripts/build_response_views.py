#!/usr/bin/env python3
"""Build independent per-response text and dense feature views."""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_ace.features import FeatureConfig
from trace_ace.feature_store import build_response_views


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/interim/cache")
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/train_transcripts",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/interim/response_views"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-char-budget", type=int, default=6000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_response_views(
        responses_path=args.cache_dir / "responses.parquet",
        session_views_path=args.cache_dir / "session_views.parquet",
        transcripts_dir=args.transcripts_dir,
        output_dir=args.output_dir,
        feature_config=FeatureConfig(context_char_budget=args.context_char_budget),
        force=args.force,
        batch_size=args.batch_size,
        progress=lambda message: print(message, flush=True),
    )
    print(f"response views: {result.status}; responses={result.response_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
