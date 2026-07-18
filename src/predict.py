#!/usr/bin/env python3
"""Run saved-ensemble inference against a runtime-format local data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_ace.ensemble import predict_test_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "models/final_ensemble_cleanroom_v02.joblib",
    )
    parser.add_argument(
        "--asset",
        type=Path,
        default=PROJECT_ROOT / "assets/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "submissions/local_submission.csv",
    )
    args = parser.parse_args()
    predictions = predict_test_directory(
        data_dir=args.data_dir,
        artifact_path=args.artifact,
        bge_asset_path=args.asset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"prediction: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
