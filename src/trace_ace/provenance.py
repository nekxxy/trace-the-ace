"""Canonical digests that bind data identities, manifests, and model assets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import pandas as pd


IDENTITY_COLUMNS = (
    "row_id",
    "response_id",
    "session_id",
    "learning_objective_id",
    "is_correct",
)


def response_identity_sha256(frame: pd.DataFrame) -> str:
    """Hash stable response/fold identities without transcript or objective text."""

    missing = [column for column in IDENTITY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"response identity columns are missing: {missing}")
    ordered = frame.loc[:, IDENTITY_COLUMNS].sort_values("row_id", kind="stable")
    if ordered["row_id"].duplicated().any() or ordered.isna().any().any():
        raise ValueError("response identities must be unique and non-missing")
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        _update_fields(digest, (str(value) for value in row))
    return digest.hexdigest()


def asset_tree_sha256(root: str | Path) -> str:
    """Hash every runtime asset path/content, excluding hub cache metadata."""

    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in directory.rglob("*")
            if ".cache" not in path.relative_to(directory).parts
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        kind = "directory" if path.is_dir() else "file"
        _update_fields(digest, (kind, relative))
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_source_sha256(root: str | Path, filenames: Iterable[str]) -> str:
    """Hash an explicit runtime-source allowlist in stable name order."""

    directory = Path(root).resolve()
    digest = hashlib.sha256()
    for name in sorted(filenames):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        _update_fields(digest, (name, file_sha256(path)))
    return digest.hexdigest()


def install_directory_with_rollback(new_directory: str | Path, destination: str | Path) -> None:
    """Install a completed generated directory while retaining rollback safety."""

    new_path = Path(new_directory)
    destination_path = Path(destination)
    if not destination_path.exists():
        os.replace(new_path, destination_path)
        return
    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.backup.", dir=destination_path.parent
        )
    )
    backup.rmdir()
    os.replace(destination_path, backup)
    try:
        os.replace(new_path, destination_path)
    except BaseException:
        os.replace(backup, destination_path)
        raise
    shutil.rmtree(backup)


def _update_fields(digest: "hashlib._Hash", fields: Iterable[str]) -> None:
    for value in fields:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
