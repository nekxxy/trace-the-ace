#!/usr/bin/env python3
"""Build a private local smoke fixture from supplied training rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        type=Path,
        default=PROJECT_ROOT / "data/raw/submission_format_5muR4s3.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data/raw/train_features_TMQTWsB.csv",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=PROJECT_ROOT / "data/raw/train_transcripts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/interim/smoke_fixture",
    )
    args = parser.parse_args()
    submission = pd.read_csv(args.format)
    features = pd.read_csv(args.features)
    if list(submission.columns) != ["response_id", "probability"]:
        raise ValueError("invalid smoke submission-format schema")
    if list(features.columns) != [
        "response_id",
        "session_id",
        "learning_objective_id",
        "learning_objective",
    ]:
        raise ValueError("invalid training-feature schema")
    if submission["response_id"].duplicated().any() or features["response_id"].duplicated().any():
        raise ValueError("response IDs must be unique")
    selected = submission[["response_id"]].merge(
        features,
        on="response_id",
        how="left",
        validate="one_to_one",
    )
    if selected.isna().any().any():
        raise ValueError("smoke IDs are not fully present in training features")
    output_dir = args.output_dir.resolve()
    safe_root = (PROJECT_ROOT / "data/interim").resolve()
    try:
        relative_output = output_dir.relative_to(safe_root)
    except ValueError as error:
        raise ValueError("smoke output must be inside data/interim") from error
    if not relative_output.parts:
        raise ValueError("smoke output cannot replace data/interim itself")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    transcript_output = output_dir / "test_transcripts"
    transcript_output.mkdir(parents=True)
    submission.to_csv(output_dir / "submission_format.csv", index=False)
    selected.loc[
        :,
        ["response_id", "session_id", "learning_objective_id", "learning_objective"],
    ].to_csv(output_dir / "test_features.csv", index=False)
    for session_id in sorted(selected["session_id"].astype(str).unique()):
        if Path(session_id).name != session_id:
            raise ValueError("session ID is not a safe transcript basename")
        source = args.transcripts / f"{session_id}.csv"
        if not source.is_file():
            raise FileNotFoundError("a required smoke transcript is missing")
        shutil.copy2(source, transcript_output / source.name)
    print("smoke fixture: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
