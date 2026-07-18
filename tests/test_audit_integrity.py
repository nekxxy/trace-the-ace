from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_integrity", PROJECT_ROOT / "scripts" / "audit_integrity.py"
)
assert SPEC is not None and SPEC.loader is not None
audit_integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_integrity)

from trace_ace.sparse_store import build_sparse_store


def _archive_fixture(root: Path) -> tuple[Path, Path, str, str]:
    archive = root / "transcripts.zip"
    extracted = root / "transcripts"
    extracted.mkdir()
    sensitive_id = "private-session-identity"
    sensitive_text = "private transcript words that must never be emitted"
    member = f"nested/{sensitive_id}.csv"
    payload = (
        "session_id,utterance_id,role,content,timestamp\n"
        f"{sensitive_id},1,student,{sensitive_text},00:00:01\n"
    ).encode()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(member, payload)
    (extracted / f"{sensitive_id}.csv").write_bytes(payload)
    return archive, extracted, sensitive_id, sensitive_text


def test_transcript_archive_audit_matches_bytes_without_leaking_values(
    tmp_path: Path,
) -> None:
    archive, extracted, sensitive_id, sensitive_text = _archive_fixture(tmp_path)

    result = audit_integrity._audit_transcript_archive(archive, extracted)
    encoded = json.dumps(result, sort_keys=True)

    # The documented official hash/count intentionally do not match this tiny
    # fixture, while all archive-to-extracted byte evidence must match.
    assert result["all_member_bytes_match"] is True
    assert result["content_tree_sha256_matches"] is True
    assert result["feature_tree_sha256_matches"] is True
    assert result["cache_tree_sha256_matches"] is True
    assert sensitive_id not in encoded
    assert sensitive_text not in encoded


def test_transcript_archive_audit_reports_mismatch_only_as_counts_and_hashes(
    tmp_path: Path,
) -> None:
    archive, extracted, sensitive_id, sensitive_text = _archive_fixture(tmp_path)
    only_file = next(extracted.glob("*.csv"))
    only_file.write_bytes(only_file.read_bytes() + b"different private bytes\n")

    result = audit_integrity._audit_transcript_archive(archive, extracted)
    encoded = json.dumps(result, sort_keys=True)

    assert result["all_member_bytes_match"] is False
    assert result["mismatched_content_count"] == 1
    assert result["content_tree_sha256_matches"] is False
    assert sensitive_id not in encoded
    assert sensitive_text not in encoded


def test_guard_never_emits_exception_text() -> None:
    sensitive = "private transcript/objective exception value"

    def fail() -> dict[str, object]:
        raise ValueError(sensitive)

    result = audit_integrity._guard(fail)
    encoded = json.dumps(result, sort_keys=True)
    assert result == {"completed": False, "passed": False}
    assert sensitive not in encoded


def _write_response_views(root: Path) -> tuple[Path, str]:
    sensitive = "private objective and transcript fixture text"
    destination = root / "data" / "interim" / "response_views"
    destination.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for row_id in range(2):
        row: dict[str, object] = {}
        for field in audit_integrity.RESPONSE_VIEWS_SCHEMA:
            if field.name == "row_id":
                value: object = row_id
            elif field.name == "response_id":
                value = f"private-response-{row_id}"
            elif field.name == "session_id":
                value = f"private-session-{row_id}"
            elif field.name == "learning_objective_id":
                value = f"private-objective-{row_id}"
            elif field.name == "learning_objective":
                value = f"{sensitive}-{row_id}"
            elif field.name == "is_correct":
                value = row_id
            elif pa.types.is_string(field.type):
                value = f"{sensitive}-{field.name}-{row_id}"
            else:
                value = 0.0
            row[field.name] = value
        rows.append(row)
    path = destination / "response_views.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=audit_integrity.RESPONSE_VIEWS_SCHEMA), path
    )
    return path, sensitive


def _copy_fingerprint_sources(root: Path, names: tuple[str, ...]) -> None:
    destination = root / "src" / "trace_ace"
    destination.mkdir(parents=True)
    for name in names:
        shutil.copyfile(PROJECT_ROOT / "src" / "trace_ace" / name, destination / name)


def test_sparse_audit_rejects_wrong_stored_dtype_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    source, sensitive = _write_response_views(tmp_path)
    _copy_fingerprint_sources(tmp_path, ("sparse_store.py", "sparse.py"))
    store = tmp_path / "data" / "processed" / "sparse_store"
    build_sparse_store(source, store, batch_size=2)

    clean = audit_integrity._audit_sparse_store(tmp_path)
    assert clean["passed"] is True
    assert clean["declared_file_count"] == 3
    assert clean["stored_array_schemas_match"] is True

    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["parts"][0]
    arrays_path = store / record["arrays"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        row_ids = arrays["row_ids"].copy()
        targets = arrays["targets"].astype(np.float32)
        dense = arrays["dense"].copy()
    np.savez_compressed(
        arrays_path, row_ids=row_ids, targets=targets, dense=dense
    )
    record["arrays_sha256"] = audit_integrity.file_sha256(arrays_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    corrupted = audit_integrity._audit_sparse_store(tmp_path)
    encoded = json.dumps(corrupted, sort_keys=True)
    assert corrupted["hash_validation_passed"] is True
    assert corrupted["targets_match"] is True
    assert corrupted["stored_array_schemas_match"] is False
    assert corrupted["passed"] is False
    assert sensitive not in encoded


def _write_semantic_store(root: Path, source: Path) -> Path:
    _copy_fingerprint_sources(
        root,
        ("semantic_store.py", "semantic.py", "config.py", "feature_store.py"),
    )
    asset_parent = root / "assets"
    asset_parent.mkdir(parents=True)
    asset = asset_parent / "bge-small-en-v1.5"
    asset.symlink_to(
        PROJECT_ROOT / "assets" / "bge-small-en-v1.5", target_is_directory=True
    )
    store = root / "data" / "processed" / "semantic_store"
    store.mkdir(parents=True)
    dimension = audit_integrity.BGE_DIMENSION
    contexts = np.zeros((2, dimension), dtype=np.float32)
    objectives = np.zeros((2, dimension), dtype=np.float32)
    contexts[:, 0] = 1.0
    objectives[0, 0] = 1.0
    objectives[1, 1] = 1.0
    objective_ids = np.asarray(
        ["private-objective-0", "private-objective-1"], dtype=np.str_
    )
    np.save(store / "context_embeddings.npy", contexts)
    np.save(store / "objective_embeddings.npy", objectives)
    np.save(store / "objective_ids.npy", objective_ids)
    np.savez_compressed(
        store / "arrays.npz",
        objective_index_by_row=np.asarray([0, 1], dtype=np.int32),
        targets=np.asarray([0, 1], dtype=np.int8),
        dense=np.zeros((2, len(audit_integrity.dense_columns())), dtype=np.float32),
    )
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
    code_root = root / "src" / "trace_ace"
    source_fingerprint = {
        "source_sha256": audit_integrity.file_sha256(source),
        "model_sha256": audit_integrity.file_sha256(asset / "model.safetensors"),
        "asset_tree_sha256": audit_integrity.asset_tree_sha256(asset),
        "response_identity_sha256": audit_integrity.response_identity_sha256(identity),
        "config": audit_integrity.ModelConfig().to_dict(),
        "code": {
            "builder_sha256": audit_integrity.file_sha256(
                code_root / "semantic_store.py"
            ),
            "config_sha256": audit_integrity.file_sha256(code_root / "config.py"),
            "encoder_sha256": audit_integrity.file_sha256(code_root / "semantic.py"),
            "feature_store_sha256": audit_integrity.file_sha256(
                code_root / "feature_store.py"
            ),
        },
        "dense_columns": list(audit_integrity.dense_columns()),
        "embedding_dimension": dimension,
        "device_independent_format": "normalized-float32",
        "build_device": "cpu",
        "encode_batch_size": 2,
        "libraries": {
            name: audit_integrity.package_version(name)
            for name in ("sentence-transformers", "transformers", "torch")
        },
    }
    output_names = (
        "context_embeddings.npy",
        "objective_embeddings.npy",
        "objective_ids.npy",
        "arrays.npz",
    )
    manifest = {
        "version": audit_integrity.SEMANTIC_STORE_VERSION,
        "source": source_fingerprint,
        "response_count": 2,
        "objective_count": 2,
        "embedding_dimension": dimension,
        "outputs": {
            name: {
                "sha256": audit_integrity.file_sha256(store / name),
                "bytes": (store / name).stat().st_size,
            }
            for name in output_names
        },
    }
    (store / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return store


def test_semantic_audit_checks_hardened_fingerprints_and_raw_array_schema(
    tmp_path: Path,
) -> None:
    source, sensitive = _write_response_views(tmp_path)
    store = _write_semantic_store(tmp_path, source)

    clean = audit_integrity._audit_semantic_store(tmp_path)
    assert clean["passed"] is True
    assert clean["declared_file_count"] == 4
    assert clean["code_fingerprints_match"] is True
    assert clean["library_fingerprints_match"] is True
    assert clean["stored_array_schemas_match"] is True

    arrays_path = store / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as arrays:
        indices = arrays["objective_index_by_row"].copy()
        targets = arrays["targets"].astype(np.float32)
        dense = arrays["dense"].copy()
    np.savez_compressed(
        arrays_path,
        objective_index_by_row=indices,
        targets=targets,
        dense=dense,
    )
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["arrays.npz"] = {
        "sha256": audit_integrity.file_sha256(arrays_path),
        "bytes": arrays_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    corrupted = audit_integrity._audit_semantic_store(tmp_path)
    encoded = json.dumps(corrupted, sort_keys=True)
    assert corrupted["hash_validation_passed"] is True
    assert corrupted["targets_match"] is True
    assert corrupted["stored_array_schemas_match"] is False
    assert corrupted["passed"] is False
    assert sensitive not in encoded
