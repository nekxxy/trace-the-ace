"""Trace the Ace offline code-execution entrypoint."""

from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from trace_ace.ensemble import predict_test_directory  # noqa: E402


ROOT = Path(__file__).resolve().parent


def main() -> None:
    predictions = predict_test_directory(
        data_dir=ROOT / "data",
        artifact_path=ROOT / "model" / "model.joblib",
        bge_asset_path=ROOT / "assets" / "bge-base-en-v1.5",
    )
    predictions.to_csv(ROOT / "submission.csv", index=False)


if __name__ == "__main__":
    main()
