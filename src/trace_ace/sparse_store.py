"""Disk-backed sparse feature parts for low-memory training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterator

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

from trace_ace.config import ModelConfig
from trace_ace.feature_store import dense_columns
from trace_ace.provenance import (
    file_sha256,
    install_directory_with_rollback,
    response_identity_sha256,
)
from trace_ace.sparse import transform_full_text, transform_role_text


SPARSE_STORE_VERSION = "trace-ace-sparse-parts-v2"


@dataclass(frozen=True)
class SparsePart:
    index: int
    row_ids: np.ndarray
    targets: np.ndarray
    dense: np.ndarray
    full: sparse.csr_matrix
    role: sparse.csr_matrix


@dataclass(frozen=True)
class SparseStoreResult:
    status: str
    path: Path
    part_count: int
    response_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_source(source: Path, config: ModelConfig, batch_size: int) -> dict[str, object]:
    identity = pq.read_table(
        source,
        columns=[
            "row_id",
            "response_id",
            "session_id",
            "learning_objective_id",
            "is_correct",
        ],
    ).to_pandas()
    ordered_ids = identity["row_id"].to_numpy()
    if not np.issubdtype(ordered_ids.dtype, np.integer):
        raise ValueError("row_id values must be integers")
    ordered_ids = np.sort(ordered_ids.astype(np.int64, copy=False))
    if not np.array_equal(ordered_ids, np.arange(len(identity), dtype=np.int64)):
        raise ValueError("row_id values must be unique and contiguous from zero")
    if not set(identity["is_correct"].unique()).issubset({0, 1, False, True}):
        raise ValueError("targets must be binary")
    return {
        "source_sha256": _sha256(source),
        "response_identity_sha256": response_identity_sha256(identity),
        "config": asdict(config),
        "batch_size": batch_size,
        "dense_columns": list(dense_columns()),
        "code": {
            "builder_sha256": file_sha256(Path(__file__)),
            "transform_sha256": file_sha256(Path(__file__).with_name("sparse.py")),
        },
    }


def build_sparse_store(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelConfig | None = None,
    batch_size: int = 128,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SparseStoreResult:
    """Hash response-view Parquet once and persist bounded CSR parts."""

    cfg = config or ModelConfig()
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _expected_source(source, cfg, batch_size)
    manifest_path = destination / "manifest.json"
    if not force and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
        if (
            manifest.get("version") == SPARSE_STORE_VERSION
            and manifest.get("source") == expected
            and _manifest_files_exist(destination, manifest, verify_hashes=True)
        ):
            if progress:
                progress("sparse store: unchanged inputs; existing parts are valid")
            return SparseStoreResult(
                "reused",
                destination,
                int(manifest["part_count"]),
                int(manifest["response_count"]),
            )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    part_records: list[dict[str, object]] = []
    response_count = 0
    try:
        parquet = pq.ParquetFile(source)
        requested = [
            "row_id",
            "is_correct",
            "full_text",
            "tutor_text",
            "student_text",
            "learning_objective",
            *dense_columns(),
        ]
        for part_index, record_batch in enumerate(
            parquet.iter_batches(batch_size=batch_size, columns=requested)
        ):
            frame = record_batch.to_pandas()
            row_ids = frame["row_id"].to_numpy(dtype=np.int64, copy=True)
            targets = frame["is_correct"].to_numpy(dtype=np.int8, copy=True)
            dense = frame.loc[:, dense_columns()].to_numpy(dtype=np.float32, copy=True)
            if len(np.unique(row_ids)) != len(row_ids) or np.any(row_ids < 0):
                raise ValueError("sparse part has invalid row_id values")
            if not set(np.unique(targets)).issubset({0, 1}):
                raise ValueError("sparse part has non-binary targets")
            if dense.shape[1] != len(dense_columns()) or not np.isfinite(dense).all():
                raise ValueError("sparse part has invalid dense features")
            full = transform_full_text(frame, cfg)
            role = transform_role_text(frame, cfg)
            expected_full = cfg.full_hash_features + cfg.objective_hash_features
            expected_role = 2 * cfg.role_hash_features + cfg.objective_hash_features
            if full.shape != (len(frame), expected_full) or role.shape != (len(frame), expected_role):
                raise ValueError("sparse feature dimensions differ from configuration")
            if full.dtype != np.float32 or role.dtype != np.float32:
                raise ValueError("sparse features must use float32")
            if not np.isfinite(full.data).all() or not np.isfinite(role.data).all():
                raise ValueError("sparse features contain non-finite values")

            stem = f"part_{part_index:05d}"
            full_name = f"{stem}.full.npz"
            role_name = f"{stem}.role.npz"
            arrays_name = f"{stem}.arrays.npz"
            sparse.save_npz(temporary / full_name, full, compressed=True)
            sparse.save_npz(temporary / role_name, role, compressed=True)
            np.savez_compressed(
                temporary / arrays_name,
                row_ids=row_ids,
                targets=targets,
                dense=dense,
            )
            part_records.append(
                {
                    "index": part_index,
                    "rows": len(frame),
                    "row_id_min": int(row_ids.min()),
                    "row_id_max": int(row_ids.max()),
                    "full_shape": list(full.shape),
                    "role_shape": list(role.shape),
                    "dense_shape": list(dense.shape),
                    "full": full_name,
                    "role": role_name,
                    "arrays": arrays_name,
                    "full_sha256": file_sha256(temporary / full_name),
                    "role_sha256": file_sha256(temporary / role_name),
                    "arrays_sha256": file_sha256(temporary / arrays_name),
                }
            )
            response_count += len(frame)
            if progress and ((part_index + 1) % 25 == 0):
                progress(f"sparse store: built {part_index + 1} parts")

        manifest = {
            "version": SPARSE_STORE_VERSION,
            "source": expected,
            "response_count": response_count,
            "part_count": len(part_records),
            "parts": part_records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            if not force:
                raise FileExistsError(
                    f"existing sparse store is invalid; rerun with --force: {destination}"
                )
        install_directory_with_rollback(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if progress:
        progress(f"sparse store: completed {response_count} responses")
    return SparseStoreResult("built", destination, len(part_records), response_count)


def _read_manifest(
    directory: Path, *, verify_hashes: bool = False
) -> dict[str, object]:
    path = directory / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid sparse store manifest: {path}") from error
    if manifest.get("version") != SPARSE_STORE_VERSION:
        raise ValueError("unsupported sparse store version")
    if not _manifest_files_exist(directory, manifest, verify_hashes=verify_hashes):
        raise ValueError("sparse store is incomplete")
    parts = manifest["parts"]
    indices = [part.get("index") for part in parts]
    if indices != list(range(len(parts))):
        raise ValueError("sparse part indices must be ordered and contiguous")
    if sum(int(part.get("rows", -1)) for part in parts) != int(manifest.get("response_count", -2)):
        raise ValueError("sparse part row totals differ from the manifest")
    names = [part[key] for part in parts for key in ("full", "role", "arrays")]
    if len(names) != len(set(names)):
        raise ValueError("sparse part filenames must be unique")
    return manifest


def _manifest_files_exist(
    directory: Path,
    manifest: dict[str, object],
    *,
    verify_hashes: bool,
) -> bool:
    parts = manifest.get("parts")
    if not isinstance(parts, list) or len(parts) != manifest.get("part_count"):
        return False
    for part in parts:
        if not isinstance(part, dict):
            return False
        for key in ("full", "role", "arrays"):
            name = part.get(key)
            expected_hash = part.get(f"{key}_sha256")
            if not isinstance(name, str) or not isinstance(expected_hash, str):
                return False
            path = directory / name
            if not path.is_file() or (
                verify_hashes and file_sha256(path) != expected_hash
            ):
                return False
    return True


def iter_sparse_parts(output_dir: str | Path) -> Iterator[SparsePart]:
    """Yield validated sparse parts in deterministic order."""

    directory = Path(output_dir).resolve()
    manifest = _read_manifest(directory)
    for record in manifest["parts"]:
        with np.load(directory / record["arrays"], allow_pickle=False) as arrays:
            row_ids = arrays["row_ids"].astype(np.int64, copy=True)
            targets = arrays["targets"].astype(np.int8, copy=True)
            dense = arrays["dense"].astype(np.float32, copy=True)
        full = sparse.load_npz(directory / record["full"]).tocsr()
        role = sparse.load_npz(directory / record["role"]).tocsr()
        if full.shape[0] != len(row_ids) or role.shape[0] != len(row_ids) or dense.shape[0] != len(row_ids):
            raise ValueError(f"row mismatch in sparse part {record['index']}")
        if list(full.shape) != record.get("full_shape") or list(role.shape) != record.get("role_shape") or list(dense.shape) != record.get("dense_shape"):
            raise ValueError(f"shape mismatch in sparse part {record['index']}")
        if np.any(row_ids < 0) or np.any(row_ids >= int(manifest["response_count"])):
            raise ValueError(f"out-of-range row_id in sparse part {record['index']}")
        yield SparsePart(int(record["index"]), row_ids, targets, dense, full, role)


def load_dense_by_row(output_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only small target/dense arrays indexed by stable row_id."""

    directory = Path(output_dir).resolve()
    manifest = _read_manifest(directory)
    row_count = int(manifest["response_count"])
    target = np.empty(row_count, dtype=np.int8)
    dense: np.ndarray | None = None
    covered = np.zeros(row_count, dtype=np.int8)
    for record in manifest["parts"]:
        with np.load(directory / record["arrays"], allow_pickle=False) as arrays:
            row_ids = arrays["row_ids"].astype(np.int64, copy=False)
            targets = arrays["targets"].astype(np.int8, copy=False)
            values = arrays["dense"].astype(np.float32, copy=False)
            if dense is None:
                dense = np.empty((row_count, values.shape[1]), dtype=np.float32)
            target[row_ids] = targets
            dense[row_ids] = values
            covered[row_ids] += 1
    if dense is None or not np.all(covered == 1):
        raise ValueError("sparse store does not cover every row_id exactly once")
    return target, dense


def validate_sparse_store(output_dir: str | Path) -> dict[str, object]:
    """Fully hash-validate a sparse store once before a training run."""

    return _read_manifest(Path(output_dir).resolve(), verify_hashes=True)
