#!/usr/bin/env python3
"""Inspect competition data files without assuming a dataset schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet", ".pq"}
SAMPLE_ROWS = 5


def format_size(size_bytes: int) -> str:
    """Return a compact human-readable file size."""
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def discover_files(data_dir: Path) -> list[Path]:
    """List direct table files without traversing a transcript corpus."""
    root = data_dir.resolve()
    discovered: list[Path] = []

    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name in {".gitkeep", "README.md"}:
            continue

        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            print(f"Skipping path outside data directory: {path}")
            continue

        discovered.append(path)

    return discovered


def summarize_subdirectories(data_dir: Path) -> None:
    """Report nested file counts without reading or printing their contents."""
    directories = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not directories:
        return

    print("Nested directories (contents not loaded):")
    for directory in directories:
        file_count = sum(1 for path in directory.rglob("*") if path.is_file())
        print(f"  - {directory.name}/ ({file_count} files)")


def read_json_file(path: Path) -> pd.DataFrame:
    """Read conventional JSON, JSON Lines, or a normalizable JSON object."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

    try:
        return pd.read_json(path)
    except ValueError:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                return pd.json_normalize(payload)
            except (OSError, TypeError, ValueError) as normalize_error:
                raise ValueError(
                    "JSON could not be read as a table, JSON Lines, or a normalizable object"
                ) from normalize_error


def load_table(path: Path) -> pd.DataFrame:
    """Load a supported file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return read_json_file(path)
    raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")


def inspect_table(path: Path, data_dir: Path, show_sample: bool) -> bool:
    """Print schema-neutral diagnostics for one table; return success status."""
    relative_path = path.relative_to(data_dir)
    print(f"\n{'=' * 80}\nFile: {relative_path}")

    try:
        frame = load_table(path)
    except Exception as error:  # Keep one unreadable file from stopping all inspection.
        print(f"Read error: {type(error).__name__}: {error}")
        return False

    summary = pd.DataFrame(
        {
            "column": [str(column) for column in frame.columns],
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing": frame.isna().sum().to_numpy(),
        }
    )

    print(f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print("Columns, data types, and missing values:")
    if summary.empty:
        print("  <no columns>")
    else:
        print(summary.to_string(index=False))

    if show_sample:
        print(f"Sample (first {min(SAMPLE_ROWS, len(frame))} rows):")
        if frame.empty:
            print("  <no rows>")
        else:
            with pd.option_context(
                "display.max_columns",
                50,
                "display.max_colwidth",
                120,
                "display.width",
                200,
            ):
                print(frame.head(SAMPLE_ROWS).to_string(index=False))

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect CSV, Parquet, and JSON files without schema assumptions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory to inspect (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Print schemas and missing counts without printing sample rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    if not data_dir.is_dir():
        print(f"Data directory does not exist: {data_dir}")
        return 1

    files = discover_files(data_dir)
    print(f"Data directory: {data_dir}")
    print(f"Top-level files found: {len(files)}")
    for path in files:
        print(f"  - {path.relative_to(data_dir)} ({format_size(path.stat().st_size)})")
    summarize_subdirectories(data_dir)

    supported = [path for path in files if path.suffix.lower() in SUPPORTED_SUFFIXES]
    unsupported = [path for path in files if path.suffix.lower() not in SUPPORTED_SUFFIXES]

    if unsupported:
        print("Unsupported files (listed but not loaded):")
        for path in unsupported:
            print(f"  - {path.relative_to(data_dir)}")

    if not supported:
        print("No CSV, Parquet, or JSON data files are available yet.")
        return 0

    successful = sum(
        inspect_table(path, data_dir, show_sample=not args.no_sample) for path in supported
    )
    print(f"\nInspection complete: {successful}/{len(supported)} supported files read successfully.")
    return 0 if successful == len(supported) else 1


if __name__ == "__main__":
    raise SystemExit(main())
