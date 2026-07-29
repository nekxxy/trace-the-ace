#!/usr/bin/env python3
"""Compute per-turn timing/latency dense features for every cached response row.

Standalone exploration tool - does NOT touch feature_store.py's production
schema. Reads each session's transcript once, reuses it across every response
row that shares that session, and writes row_id -> timing feature vector to a
parquet file for downstream CV ablation (see graph_ablation_cv.py, which is
generic enough to reuse directly by pointing --graph-features at this output).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trace_ace.timing_features import TIMING_DENSE_FEATURE_NAMES, extract_timing_features
from trace_ace.io import read_transcript


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        default=PROJECT_ROOT / "data/interim/cache/responses.parquet",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/train_transcripts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/interim/timing_features/timing_dense.parquet",
    )
    args = parser.parse_args()

    responses = pd.read_parquet(args.responses).sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(responses["row_id"].to_numpy(), np.arange(len(responses))):
        raise ValueError("responses row_id must be contiguous")

    n = len(responses)
    vectors = np.zeros((n, len(TIMING_DENSE_FEATURE_NAMES)), dtype=np.float64)

    start = time.time()
    by_session = responses.groupby("session_id", sort=False)
    processed_sessions = 0
    processed_rows = 0
    for session_id, group in by_session:
        transcript_path = args.transcripts_dir / f"{session_id}.csv"
        transcript = read_transcript(transcript_path, expected_session_id=session_id)
        for row_id, objective in zip(group["row_id"], group["learning_objective"]):
            result = extract_timing_features(transcript, objective)
            vectors[row_id] = result.dense
            processed_rows += 1
        processed_sessions += 1
        if processed_sessions % 2000 == 0:
            elapsed = time.time() - start
            print(
                f"sessions={processed_sessions} rows={processed_rows}/{n} elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - start
    print(
        f"DONE sessions={processed_sessions} rows={processed_rows} elapsed={elapsed:.1f}s",
        flush=True,
    )

    if not np.isfinite(vectors).all():
        raise ValueError("timing feature matrix contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(vectors, columns=list(TIMING_DENSE_FEATURE_NAMES))
    frame.insert(0, "row_id", np.arange(n, dtype=np.int64))
    frame.to_parquet(args.output, index=False)
    print(f"wrote {args.output} shape={frame.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
