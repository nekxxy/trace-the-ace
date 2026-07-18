"""Build and load disk-backed BGE embeddings without retaining text in RAM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

from trace_ace.config import BGE_DIMENSION, ModelConfig
from trace_ace.feature_store import dense_columns
from trace_ace.provenance import (
    asset_tree_sha256,
    file_sha256,
    install_directory_with_rollback,
    response_identity_sha256,
)
from trace_ace.semantic import encode_texts, load_sentence_transformer


SEMANTIC_STORE_VERSION = "trace-ace-bge-context-v2"


@dataclass(frozen=True)
class SemanticStoreResult:
    status: str
    path: Path
    response_count: int
    objective_count: int


@dataclass(frozen=True)
class SemanticArrays:
    context_embeddings: np.ndarray
    objective_embeddings: np.ndarray
    objective_ids: np.ndarray
    objective_index_by_row: np.ndarray
    targets: np.ndarray
    dense: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_semantic_store(
    source_path: str | Path,
    asset_path: str | Path,
    output_dir: str | Path,
    *,
    config: ModelConfig | None = None,
    device: str = "cpu",
    encode_batch_size: int | None = None,
    read_batch_size: int = 128,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SemanticStoreResult:
    """Encode objective-conditioned contexts locally into row-indexed arrays."""

    cfg = config or ModelConfig()
    source = Path(source_path).resolve()
    asset = Path(asset_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not asset.is_dir():
        raise FileNotFoundError(asset)
    if read_batch_size < 1:
        raise ValueError("read_batch_size must be positive")
    if encode_batch_size is not None and encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    batch_size = encode_batch_size or (
        cfg.bge_batch_size_gpu if device.startswith("cuda") else cfg.bge_batch_size_cpu
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_weights = asset / "model.safetensors"
    if not model_weights.is_file():
        raise FileNotFoundError(model_weights)
    identity_digest_frame = pq.read_table(
        source,
        columns=[
            "row_id",
            "response_id",
            "session_id",
            "learning_objective_id",
            "is_correct",
        ],
    ).to_pandas()
    source_fingerprint = {
        "source_sha256": _sha256(source),
        "model_sha256": _sha256(model_weights),
        "asset_tree_sha256": asset_tree_sha256(asset),
        "response_identity_sha256": response_identity_sha256(identity_digest_frame),
        "config": asdict(cfg),
        "code": {
            "builder_sha256": file_sha256(Path(__file__)),
            "config_sha256": file_sha256(Path(__file__).with_name("config.py")),
            "encoder_sha256": file_sha256(Path(__file__).with_name("semantic.py")),
            "feature_store_sha256": file_sha256(
                Path(__file__).with_name("feature_store.py")
            ),
        },
        "dense_columns": list(dense_columns()),
        "embedding_dimension": BGE_DIMENSION,
        "device_independent_format": "normalized-float32",
        "build_device": device,
        "encode_batch_size": batch_size,
        "libraries": {
            name: package_version(name)
            for name in ("sentence-transformers", "transformers", "torch")
        },
    }
    manifest_path = destination / "manifest.json"
    if not force and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
        if (
            manifest.get("version") == SEMANTIC_STORE_VERSION
            and manifest.get("source") == source_fingerprint
            and _semantic_files_valid(destination, manifest)
        ):
            if progress:
                progress("semantic store: unchanged inputs; existing embeddings are valid")
            return SemanticStoreResult(
                "reused",
                destination,
                int(manifest["response_count"]),
                int(manifest["objective_count"]),
            )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        parquet = pq.ParquetFile(source)
        row_count = parquet.metadata.num_rows
        identity = pq.read_table(
            source,
            columns=[
                "row_id",
                "learning_objective_id",
                "learning_objective",
                "is_correct",
                *dense_columns(),
            ],
        ).to_pandas().sort_values("row_id", kind="stable")
        row_ids = identity["row_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(row_ids, np.arange(row_count, dtype=np.int64)):
            raise ValueError("response views must contain contiguous row_id values")
        objective_table = (
            identity[["learning_objective_id", "learning_objective"]]
            .drop_duplicates()
            .sort_values("learning_objective_id", kind="stable")
            .reset_index(drop=True)
        )
        if objective_table["learning_objective_id"].duplicated().any():
            raise ValueError("one objective ID maps to multiple objective texts")
        objective_ids = np.asarray(
            objective_table["learning_objective_id"].astype(str).tolist(),
            dtype=np.str_,
        )
        objective_lookup = {value: index for index, value in enumerate(objective_ids)}
        objective_indices = (
            identity["learning_objective_id"].astype(str).map(objective_lookup).to_numpy(dtype=np.int32)
        )
        targets = identity["is_correct"].to_numpy(dtype=np.int8)
        if not set(np.unique(targets)).issubset({0, 1}):
            raise ValueError("targets must be binary")
        dense = identity.loc[:, dense_columns()].to_numpy(dtype=np.float32)
        if not np.isfinite(dense).all():
            raise ValueError("dense features contain non-finite values")

        encoder = load_sentence_transformer(
            asset,
            max_seq_length=cfg.bge_max_seq_length,
            device=device,
        )
        objective_embeddings = encode_texts(
            encoder,
            objective_table["learning_objective"].astype(str).tolist(),
            batch_size=batch_size,
        )
        if objective_embeddings.shape != (len(objective_ids), BGE_DIMENSION):
            raise ValueError("unexpected objective embedding shape")
        np.save(temporary / "objective_embeddings.npy", objective_embeddings)
        np.save(temporary / "objective_ids.npy", objective_ids)

        context_path = temporary / "context_embeddings.npy"
        contexts = np.lib.format.open_memmap(
            context_path,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, BGE_DIMENSION),
        )
        covered = np.zeros(row_count, dtype=np.int8)
        encoded_count = 0
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=["row_id", "objective_context"],
        ):
            frame = batch.to_pandas()
            ids = frame["row_id"].to_numpy(dtype=np.int64)
            encoded = encode_texts(
                encoder,
                frame["objective_context"].astype(str).tolist(),
                batch_size=batch_size,
            )
            if encoded.shape[1] != BGE_DIMENSION:
                raise ValueError("unexpected context embedding dimension")
            contexts[ids] = encoded
            if np.any(ids < 0) or np.any(ids >= row_count):
                raise ValueError("semantic batch contains out-of-range row_id")
            covered[ids] += 1
            encoded_count += len(ids)
            if progress and encoded_count % 1024 < len(ids):
                progress(f"semantic store: encoded {encoded_count}/{row_count} responses")
        contexts.flush()
        del contexts, encoder
        gc.collect()
        if not np.all(covered == 1):
            raise ValueError("semantic encoding must cover every row exactly once")
        np.savez_compressed(
            temporary / "arrays.npz",
            objective_index_by_row=objective_indices,
            targets=targets,
            dense=dense,
        )
        manifest = {
            "version": SEMANTIC_STORE_VERSION,
            "source": source_fingerprint,
            "response_count": row_count,
            "objective_count": len(objective_ids),
            "embedding_dimension": BGE_DIMENSION,
            "outputs": {
                name: {
                    "sha256": file_sha256(temporary / name),
                    "bytes": (temporary / name).stat().st_size,
                }
                for name in (
                    "context_embeddings.npy",
                    "objective_embeddings.npy",
                    "objective_ids.npy",
                    "arrays.npz",
                )
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            if not force:
                raise FileExistsError(
                    f"existing semantic store is invalid; rerun with --force: {destination}"
                )
        install_directory_with_rollback(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if progress:
        progress(f"semantic store: completed {row_count} responses")
    return SemanticStoreResult("built", destination, row_count, len(objective_ids))


def _semantic_files_valid(directory: Path, manifest: dict[str, object]) -> bool:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return False
    required = (
        "context_embeddings.npy",
        "objective_embeddings.npy",
        "objective_ids.npy",
        "arrays.npz",
    )
    for name in required:
        record = outputs.get(name)
        path = directory / name
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != file_sha256(path)
        ):
            return False
    return True


def load_semantic_arrays(
    output_dir: str | Path,
    *,
    mmap_mode: str | None = "r",
) -> SemanticArrays:
    directory = Path(output_dir).resolve()
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("invalid semantic store manifest") from error
    if manifest.get("version") != SEMANTIC_STORE_VERSION or not _semantic_files_valid(directory, manifest):
        raise ValueError("semantic store is incomplete or unsupported")
    contexts = np.load(directory / "context_embeddings.npy", mmap_mode=mmap_mode, allow_pickle=False)
    objectives = np.load(directory / "objective_embeddings.npy", mmap_mode=mmap_mode, allow_pickle=False)
    objective_ids = np.load(directory / "objective_ids.npy", allow_pickle=False)
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        indices = arrays["objective_index_by_row"].astype(np.int32, copy=True)
        targets = arrays["targets"].astype(np.int8, copy=True)
        dense = arrays["dense"].astype(np.float32, copy=True)
    if contexts.shape[0] != len(targets) or len(indices) != len(targets):
        raise ValueError("semantic store row counts differ")
    return SemanticArrays(contexts, objectives, objective_ids, indices, targets, dense)


def load_semantic_manifest(output_dir: str | Path) -> dict[str, object]:
    directory = Path(output_dir).resolve()
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("invalid semantic store manifest") from error
    if manifest.get("version") != SEMANTIC_STORE_VERSION or not _semantic_files_valid(
        directory, manifest
    ):
        raise ValueError("semantic store is incomplete or unsupported")
    return manifest
