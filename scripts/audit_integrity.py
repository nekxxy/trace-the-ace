#!/usr/bin/env python3
"""Read-only, non-leaking integrity audit for clean-room training artifacts.

The single stdout value is JSON containing only booleans, counts, schemas, and
SHA-256 digests.  In particular, this module never emits response/session IDs,
archive member names, objectives, transcript text, paths from manifests, or
exception messages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trace_ace.cache import (  # noqa: E402
    CACHE_VERSION,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    RESPONSES_SCHEMA,
    SESSION_VIEWS_SCHEMA,
)
from trace_ace.config import BGE_DIMENSION, ModelConfig  # noqa: E402
from trace_ace.feature_store import (  # noqa: E402
    FEATURE_STORE_VERSION,
    RESPONSE_VIEWS_SCHEMA,
    dense_columns,
)
from trace_ace.features import DENSE_FEATURE_NAMES, FeatureConfig  # noqa: E402
from trace_ace.io import read_transcript  # noqa: E402
from trace_ace.provenance import (  # noqa: E402
    asset_tree_sha256,
    file_sha256,
    response_identity_sha256,
)
from trace_ace.semantic_store import (  # noqa: E402
    SEMANTIC_STORE_VERSION,
    load_semantic_arrays,
    load_semantic_manifest,
)
from trace_ace.sparse_store import (  # noqa: E402
    SPARSE_STORE_VERSION,
    iter_sparse_parts,
    load_dense_by_row,
    validate_sparse_store,
)


AUDIT_VERSION = 1
CHUNK_SIZE = 1024 * 1024
DOCUMENTED_TRANSCRIPT_ROW_COUNT = 6_139_854
DOCUMENTED_TRANSCRIPT_CONTENT_CHARS = 416_437_335

# Mirrors docs/data_inventory.md.  The documentation itself is checked for
# these exact pairs so a documentation edit cannot silently change the audit.
SOURCE_SPECS: Mapping[str, Mapping[str, object]] = {
    "train_features": {
        "filename": "train_features_TMQTWsB.csv",
        "sha256": "71bea3abb76a1cff5e1eaa75b9cbcfaf26d0419f6274b83a199ed520047a5063",
        "row_count": 35_072,
        "schema": tuple(FEATURE_COLUMNS),
    },
    "train_labels": {
        "filename": "train_labels_44ujmj2.csv",
        "sha256": "d98ee4389e5cde3f66d6d15b7b574261024a80e405958eca333d3c1921fd65b9",
        "row_count": 35_072,
        "schema": tuple(LABEL_COLUMNS),
    },
    "smoke_submission_format": {
        "filename": "submission_format_5muR4s3.csv",
        "sha256": "ec4a4b1debdf05bce37c95bb3884c058c8807a8b12e1bd7d59f03ddcca560e6e",
        "row_count": 100,
        "schema": ("response_id", "probability"),
    },
    "normal_submission_format": {
        "filename": "submission_format_ZQLcKx7.csv",
        "sha256": "957bc8b1fe9a69f74ee1bdc63edcc389810b191247645558396ea437b26ac0fa",
        "row_count": 10_508,
        "schema": ("response_id", "probability"),
    },
    "transcript_archive": {
        "filename": "train_transcripts.zip",
        "sha256": "e685b85b04694e130c25b17d09cdd1892fbda5e9fa685e98b2300114b915aa2d",
        "file_count": 22_821,
    },
}


def _sha256_fields(fields: Iterable[object], *, domain: bytes = b"") -> str:
    digest = hashlib.sha256(domain)
    for value in fields:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256_schema(schema: pa.Schema) -> str:
    return hashlib.sha256(str(schema).encode("utf-8")).hexdigest()


def _sha256_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_by: Sequence[str],
    domain: bytes,
) -> str:
    ordered = frame.sort_values(list(sort_by), kind="stable")
    digest = hashlib.sha256(domain)
    for row in ordered.loc[:, list(columns)].itertuples(index=False, name=None):
        for value in row:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _sha256_string_set(values: Iterable[object], *, domain: bytes) -> str:
    return _sha256_fields(sorted(str(value) for value in values), domain=domain)


def _safe_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: object, default: int = -1) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest is not an object")
    return value


def _guard(check: Callable[[], dict[str, object]]) -> dict[str, object]:
    """Contain failures without leaking exception strings or unsafe values."""

    try:
        result = check()
    except Exception:
        return {"completed": False, "passed": False}
    result["completed"] = True
    result["passed"] = bool(result.get("passed", False))
    return result


def _read_csv_contract(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    if list(frame.columns) != list(expected_columns):
        raise ValueError("unexpected CSV schema")
    return frame


def _audit_documented_sources(root: Path) -> dict[str, object]:
    source_root = root / "data" / "source"
    documentation = root / "docs" / "data_inventory.md"
    documentation_text = documentation.read_text(encoding="utf-8")
    records: dict[str, object] = {}
    all_passed = True
    documented_pairs_present = True
    for logical_name, spec in SOURCE_SPECS.items():
        filename = str(spec["filename"])
        expected_sha256 = str(spec["sha256"])
        documented_pairs_present &= (
            f"`{filename}`" in documentation_text
            and f"`{expected_sha256}`" in documentation_text
        )
        path = source_root / filename
        exists = path.is_file()
        actual_sha256 = file_sha256(path) if exists else hashlib.sha256(b"").hexdigest()
        record: dict[str, object] = {
            "exists": exists,
            "sha256": actual_sha256,
            "sha256_matches": exists and actual_sha256 == expected_sha256,
        }
        if "row_count" in spec:
            expected_schema = tuple(str(value) for value in spec["schema"])  # type: ignore[index]
            if exists:
                frame = pd.read_csv(path, dtype="string", keep_default_na=False)
                row_count = len(frame)
                schema_matches = tuple(frame.columns) == expected_schema
            else:
                row_count = 0
                schema_matches = False
            record.update(
                {
                    "row_count": row_count,
                    "row_count_matches": row_count == int(spec["row_count"]),
                    "schema": list(expected_schema),
                    "schema_matches": schema_matches,
                }
            )
        elif "file_count" in spec:
            record["documented_file_count"] = int(spec["file_count"])
        record_passed = bool(
            record["exists"]
            and record["sha256_matches"]
            and record.get("row_count_matches", True)
            and record.get("schema_matches", True)
        )
        record["passed"] = record_passed
        all_passed &= record_passed
        records[logical_name] = record
    return {
        "documentation_sha256": file_sha256(documentation),
        "documented_pairs_present": documented_pairs_present,
        "sources": records,
        "passed": all_passed and documented_pairs_present,
    }


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(CHUNK_SIZE)
            right_chunk = right_stream.read(CHUNK_SIZE)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _audit_source_raw_csv_parity(root: Path) -> dict[str, object]:
    source_root = root / "data" / "source"
    raw_root = root / "data" / "raw"
    records: dict[str, object] = {}
    passed = True
    for logical_name, spec in SOURCE_SPECS.items():
        if not str(spec["filename"]).endswith(".csv"):
            continue
        source = source_root / str(spec["filename"])
        raw = raw_root / str(spec["filename"])
        source_exists = source.is_file()
        raw_exists = raw.is_file()
        source_hash = file_sha256(source) if source_exists else hashlib.sha256(b"").hexdigest()
        raw_hash = file_sha256(raw) if raw_exists else hashlib.sha256(b"").hexdigest()
        byte_count = source.stat().st_size if source_exists else 0
        bytes_match = (
            source_exists and raw_exists and source_hash == raw_hash and _files_equal(source, raw)
        )
        record = {
            "source_exists": source_exists,
            "raw_exists": raw_exists,
            "byte_count": byte_count,
            "source_sha256": source_hash,
            "raw_sha256": raw_hash,
            "bytes_match": bytes_match,
        }
        record["passed"] = bool(source_exists and raw_exists and bytes_match)
        passed &= bool(record["passed"])
        records[logical_name] = record
    return {"tables": records, "passed": passed}


def _update_named_file_prefix(digest: "hashlib._Hash", name: str) -> None:
    encoded = name.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _audit_transcript_archive(archive: Path, extracted_root: Path) -> dict[str, object]:
    raw_files = sorted(extracted_root.glob("*.csv"), key=lambda path: path.name)
    raw_by_name = {path.name: path for path in raw_files}

    zip_feature_tree = hashlib.sha256()
    raw_feature_tree = hashlib.sha256()
    zip_cache_tree = hashlib.sha256(b"trace-ace-transcript-tree-v1\0")
    raw_cache_tree = hashlib.sha256(b"trace-ace-transcript-tree-v1\0")
    zip_content_tree = hashlib.sha256(b"trace-ace-transcript-content-v1\0")
    raw_content_tree = hashlib.sha256(b"trace-ace-transcript-content-v1\0")

    with zipfile.ZipFile(archive, "r") as bundle:
        file_infos = [info for info in bundle.infolist() if not info.is_dir()]
        csv_infos = [
            info for info in file_infos if PurePosixPath(info.filename).suffix.casefold() == ".csv"
        ]
        basenames = [PurePosixPath(info.filename).name for info in csv_infos]
        counts = Counter(basenames)
        unique_info = {
            PurePosixPath(info.filename).name: info
            for info in csv_infos
            if counts[PurePosixPath(info.filename).name] == 1
        }
        unsafe_member_count = sum(
            PurePosixPath(info.filename).is_absolute()
            or ".." in PurePosixPath(info.filename).parts
            or not PurePosixPath(info.filename).name
            for info in csv_infos
        )

        zip_total_bytes = 0
        raw_total_bytes = 0
        mismatched_content_count = 0
        missing_names = set(unique_info) - set(raw_by_name)
        unexpected_names = set(raw_by_name) - set(unique_info)

        for name in sorted(unique_info):
            info = unique_info[name]
            _update_named_file_prefix(zip_feature_tree, name)
            _update_named_file_prefix(zip_content_tree, name)
            _update_named_file_prefix(zip_cache_tree, name)
            zip_file_digest = hashlib.sha256()
            raw_path = raw_by_name.get(name)
            raw_file_digest = hashlib.sha256()
            same = raw_path is not None
            with bundle.open(info, "r") as zip_stream:
                if raw_path is None:
                    while True:
                        chunk = zip_stream.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        zip_file_digest.update(chunk)
                        zip_feature_tree.update(chunk)
                        zip_total_bytes += len(chunk)
                else:
                    _update_named_file_prefix(raw_feature_tree, name)
                    _update_named_file_prefix(raw_content_tree, name)
                    _update_named_file_prefix(raw_cache_tree, name)
                    with raw_path.open("rb") as raw_stream:
                        while True:
                            zip_chunk = zip_stream.read(CHUNK_SIZE)
                            raw_chunk = raw_stream.read(CHUNK_SIZE)
                            if not zip_chunk and not raw_chunk:
                                break
                            same &= zip_chunk == raw_chunk
                            zip_file_digest.update(zip_chunk)
                            raw_file_digest.update(raw_chunk)
                            zip_feature_tree.update(zip_chunk)
                            raw_feature_tree.update(raw_chunk)
                            zip_total_bytes += len(zip_chunk)
                            raw_total_bytes += len(raw_chunk)
            zip_digest = zip_file_digest.hexdigest()
            _update_named_file_prefix(zip_content_tree, zip_digest)
            zip_cache_tree.update(bytes.fromhex(zip_digest))
            if raw_path is not None:
                raw_digest = raw_file_digest.hexdigest()
                _update_named_file_prefix(raw_content_tree, raw_digest)
                raw_cache_tree.update(bytes.fromhex(raw_digest))
                same &= zip_digest == raw_digest
            if not same:
                mismatched_content_count += 1

        # Files absent from the archive still participate in extracted-tree
        # evidence, keeping the two aggregate digests independently meaningful.
        for name in sorted(unexpected_names):
            raw_path = raw_by_name[name]
            _update_named_file_prefix(raw_feature_tree, name)
            _update_named_file_prefix(raw_content_tree, name)
            _update_named_file_prefix(raw_cache_tree, name)
            raw_file_digest = hashlib.sha256()
            with raw_path.open("rb") as raw_stream:
                while True:
                    chunk = raw_stream.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    raw_file_digest.update(chunk)
                    raw_feature_tree.update(chunk)
                    raw_total_bytes += len(chunk)
            raw_digest = raw_file_digest.hexdigest()
            _update_named_file_prefix(raw_content_tree, raw_digest)
            raw_cache_tree.update(bytes.fromhex(raw_digest))

    documented_count = int(SOURCE_SPECS["transcript_archive"]["file_count"])
    archive_hash = file_sha256(archive)
    counts_match = len(csv_infos) == len(raw_files) == documented_count
    bytes_match = (
        not missing_names
        and not unexpected_names
        and mismatched_content_count == 0
        and zip_total_bytes == raw_total_bytes
    )
    structure_valid = (
        len(file_infos) == len(csv_infos)
        and not any(count > 1 for count in counts.values())
        and unsafe_member_count == 0
    )
    return {
        "archive_sha256": archive_hash,
        "archive_sha256_matches_documentation": archive_hash
        == SOURCE_SPECS["transcript_archive"]["sha256"],
        "archive_file_count": len(csv_infos),
        "extracted_file_count": len(raw_files),
        "documented_file_count": documented_count,
        "file_counts_match": counts_match,
        "archive_non_csv_file_count": len(file_infos) - len(csv_infos),
        "duplicate_basename_count": sum(count - 1 for count in counts.values() if count > 1),
        "unsafe_member_count": unsafe_member_count,
        "missing_extracted_count": len(missing_names),
        "unexpected_extracted_count": len(unexpected_names),
        "mismatched_content_count": mismatched_content_count,
        "archive_uncompressed_bytes": zip_total_bytes,
        "extracted_bytes": raw_total_bytes,
        "all_member_bytes_match": bytes_match,
        "archive_content_tree_sha256": zip_content_tree.hexdigest(),
        "extracted_content_tree_sha256": raw_content_tree.hexdigest(),
        "content_tree_sha256_matches": zip_content_tree.digest() == raw_content_tree.digest(),
        "archive_feature_tree_sha256": zip_feature_tree.hexdigest(),
        "extracted_feature_tree_sha256": raw_feature_tree.hexdigest(),
        "feature_tree_sha256_matches": zip_feature_tree.digest() == raw_feature_tree.digest(),
        "archive_cache_tree_sha256": zip_cache_tree.hexdigest(),
        "extracted_cache_tree_sha256": raw_cache_tree.hexdigest(),
        "cache_tree_sha256_matches": zip_cache_tree.digest() == raw_cache_tree.digest(),
        "passed": bool(
            archive_hash == SOURCE_SPECS["transcript_archive"]["sha256"]
            and counts_match
            and structure_valid
            and bytes_match
            and zip_content_tree.digest() == raw_content_tree.digest()
            and zip_feature_tree.digest() == raw_feature_tree.digest()
            and zip_cache_tree.digest() == raw_cache_tree.digest()
        ),
    }


def _parquet_evidence(path: Path, expected_schema: pa.Schema) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    actual_schema = parquet.schema_arrow
    return {
        "sha256": file_sha256(path),
        "row_count": parquet.metadata.num_rows,
        "schema_sha256": _sha256_schema(actual_schema),
        "expected_schema_sha256": _sha256_schema(expected_schema),
        "schema_matches": actual_schema == expected_schema,
    }


def _audit_cache(root: Path, transcript_evidence: Mapping[str, object]) -> dict[str, object]:
    cache_dir = root / "data" / "interim" / "cache"
    responses_path = cache_dir / "responses.parquet"
    sessions_path = cache_dir / "session_views.parquet"
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = _load_json_object(manifest_path)
    sources = _safe_dict(manifest.get("sources"))
    code = _safe_dict(sources.get("code"))
    outputs = _safe_dict(manifest.get("outputs"))
    response_manifest = _safe_dict(outputs.get("responses"))
    session_manifest = _safe_dict(outputs.get("session_views"))

    responses = _parquet_evidence(responses_path, RESPONSES_SCHEMA)
    sessions = _parquet_evidence(sessions_path, SESSION_VIEWS_SCHEMA)
    expected_builder_hash = file_sha256(root / "src" / "trace_ace" / "cache.py")
    expected_reader_hash = file_sha256(root / "src" / "trace_ace" / "io.py")
    source_features_hash = file_sha256(
        root / "data" / "raw" / str(SOURCE_SPECS["train_features"]["filename"])
    )
    source_labels_hash = file_sha256(
        root / "data" / "raw" / str(SOURCE_SPECS["train_labels"]["filename"])
    )
    transcript_source = _safe_dict(sources.get("transcripts"))
    expected_transcript_hash = str(
        transcript_evidence.get("extracted_cache_tree_sha256", "")
    )

    response_output_matches = bool(
        response_manifest.get("path") == "responses.parquet"
        and response_manifest.get("sha256") == responses["sha256"]
        and _safe_int(response_manifest.get("row_count")) == responses["row_count"]
        and response_manifest.get("schema") == str(RESPONSES_SCHEMA)
        and responses["schema_matches"]
    )
    session_output_matches = bool(
        session_manifest.get("path") == "session_views.parquet"
        and session_manifest.get("sha256") == sessions["sha256"]
        and _safe_int(session_manifest.get("row_count")) == sessions["row_count"]
        and session_manifest.get("schema") == str(SESSION_VIEWS_SCHEMA)
        and sessions["schema_matches"]
    )
    source_fingerprints_match = bool(
        _safe_dict(sources.get("features")).get("sha256") == source_features_hash
        and _safe_dict(sources.get("labels")).get("sha256") == source_labels_hash
        and transcript_source.get("sha256") == expected_transcript_hash
        and _safe_int(transcript_source.get("file_count"))
        == transcript_evidence.get("extracted_file_count")
    )
    code_fingerprints_match = bool(
        code.get("builder_sha256") == expected_builder_hash
        and code.get("reader_sha256") == expected_reader_hash
    )
    version_matches = manifest.get("cache_version") == CACHE_VERSION
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "version_matches": version_matches,
        "source_features_sha256": source_features_hash,
        "source_labels_sha256": source_labels_hash,
        "source_transcript_tree_sha256": expected_transcript_hash,
        "source_fingerprints_match": source_fingerprints_match,
        "builder_sha256": expected_builder_hash,
        "reader_sha256": expected_reader_hash,
        "code_fingerprints_match": code_fingerprints_match,
        "responses": {**responses, "manifest_matches": response_output_matches},
        "session_views": {**sessions, "manifest_matches": session_output_matches},
        "passed": bool(
            version_matches
            and source_fingerprints_match
            and code_fingerprints_match
            and response_output_matches
            and session_output_matches
        ),
    }


def _expected_source_responses(root: Path) -> pd.DataFrame:
    raw = root / "data" / "raw"
    features = _read_csv_contract(
        raw / str(SOURCE_SPECS["train_features"]["filename"]), FEATURE_COLUMNS
    )
    labels = _read_csv_contract(
        raw / str(SOURCE_SPECS["train_labels"]["filename"]), LABEL_COLUMNS
    )
    if features["response_id"].duplicated().any() or labels["response_id"].duplicated().any():
        raise ValueError("duplicate response identity")
    if set(features["response_id"]) != set(labels["response_id"]):
        raise ValueError("response coverage mismatch")
    numeric = pd.to_numeric(labels["is_correct"], errors="raise")
    if not numeric.isin([0, 1]).all():
        raise ValueError("invalid labels")
    labels = labels.copy()
    labels["is_correct"] = numeric.astype("int8")
    merged = features.merge(labels, on="response_id", validate="one_to_one", sort=False)
    merged = merged.sort_values("response_id", kind="stable").reset_index(drop=True)
    merged.insert(0, "row_id", np.arange(len(merged), dtype=np.int64))
    return merged


def _read_response_cache(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    if table.schema != RESPONSES_SCHEMA:
        raise ValueError("unexpected response-cache schema")
    return table.to_pandas()


def _audit_response_parity(root: Path) -> dict[str, object]:
    expected = _expected_source_responses(root)
    cached = _read_response_cache(root / "data" / "interim" / "cache" / "responses.parquet")
    expected_identity = response_identity_sha256(expected)
    cached_identity = response_identity_sha256(cached)
    expected_labels = _sha256_frame(
        expected,
        ("row_id", "response_id", "is_correct"),
        sort_by=("row_id",),
        domain=b"trace-ace-response-labels-v1\0",
    )
    cached_labels = _sha256_frame(
        cached,
        ("row_id", "response_id", "is_correct"),
        sort_by=("row_id",),
        domain=b"trace-ace-response-labels-v1\0",
    )
    expected_objectives = _sha256_frame(
        expected,
        ("row_id", "learning_objective"),
        sort_by=("row_id",),
        domain=b"trace-ace-response-objective-text-v1\0",
    )
    cached_objectives = _sha256_frame(
        cached,
        ("row_id", "learning_objective"),
        sort_by=("row_id",),
        domain=b"trace-ace-response-objective-text-v1\0",
    )
    cached_row_ids = cached["row_id"].to_numpy(dtype=np.int64)
    contiguous = np.array_equal(
        np.sort(cached_row_ids), np.arange(len(cached), dtype=np.int64)
    )
    passed = bool(
        len(expected) == len(cached)
        and contiguous
        and expected_identity == cached_identity
        and expected_labels == cached_labels
        and expected_objectives == cached_objectives
    )
    return {
        "source_row_count": len(expected),
        "cache_row_count": len(cached),
        "row_ids_contiguous": contiguous,
        "source_identity_sha256": expected_identity,
        "cache_identity_sha256": cached_identity,
        "identity_matches": expected_identity == cached_identity,
        "source_label_sha256": expected_labels,
        "cache_label_sha256": cached_labels,
        "labels_match": expected_labels == cached_labels,
        "source_objective_text_sha256": expected_objectives,
        "cache_objective_text_sha256": cached_objectives,
        "objective_text_matches": expected_objectives == cached_objectives,
        "passed": passed,
    }


def _identity_set_evidence(values: Sequence[object], *, domain: bytes) -> dict[str, object]:
    normalized = [str(value) for value in values]
    unique = set(normalized)
    return {
        "row_count": len(normalized),
        "unique_count": len(unique),
        "duplicate_count": len(normalized) - len(unique),
        "missing_count": sum(not value for value in normalized),
        "identity_set_sha256": _sha256_string_set(unique, domain=domain),
    }


def _audit_session_coverage(root: Path) -> dict[str, object]:
    raw = root / "data" / "raw"
    features = _read_csv_contract(
        raw / str(SOURCE_SPECS["train_features"]["filename"]), FEATURE_COLUMNS
    )
    source_sessions = features["session_id"].astype(str).drop_duplicates().tolist()
    transcript_paths = sorted((raw / "train_transcripts").glob("*.csv"))
    transcript_sessions = [path.stem for path in transcript_paths]
    strict_reader_invalid_count = 0
    strict_reader_row_count = 0
    strict_reader_content_chars = 0
    strict_reader_sessions: list[str] = []
    for transcript_path in transcript_paths:
        try:
            transcript = read_transcript(
                transcript_path, expected_session_id=transcript_path.stem
            )
        except Exception:
            strict_reader_invalid_count += 1
            continue
        strict_reader_row_count += len(transcript)
        strict_reader_content_chars += int(
            transcript["content"].astype("string").str.len().sum()
        )
        strict_reader_sessions.append(str(transcript["session_id"].iloc[0]))
    responses = pq.read_table(
        root / "data" / "interim" / "cache" / "responses.parquet",
        columns=["session_id"],
    ).column("session_id").to_pylist()
    response_sessions = list(dict.fromkeys(str(value) for value in responses))
    session_views = pq.read_table(
        root / "data" / "interim" / "cache" / "session_views.parquet",
        columns=["session_id"],
    ).column("session_id").to_pylist()

    domain = b"trace-ace-session-identity-set-v1\0"
    evidence = {
        "source_features": _identity_set_evidence(source_sessions, domain=domain),
        "extracted_transcripts": _identity_set_evidence(transcript_sessions, domain=domain),
        "strict_reader_transcripts": _identity_set_evidence(
            strict_reader_sessions, domain=domain
        ),
        "cache_responses": _identity_set_evidence(response_sessions, domain=domain),
        "cache_session_views": _identity_set_evidence(session_views, domain=domain),
    }
    sets = [
        set(str(value) for value in source_sessions),
        set(str(value) for value in transcript_sessions),
        set(str(value) for value in strict_reader_sessions),
        set(str(value) for value in response_sessions),
        set(str(value) for value in session_views),
    ]
    all_match = all(values == sets[0] for values in sets[1:])
    session_views_unique = len(session_views) == len(sets[-1])
    no_missing = all(
        _safe_int(_safe_dict(record).get("missing_count")) == 0
        for record in evidence.values()
    )
    return {
        "sets": evidence,
        "all_identity_sets_match": all_match,
        "session_views_unique": session_views_unique,
        "no_missing_identities": no_missing,
        "strict_reader_invalid_count": strict_reader_invalid_count,
        "strict_reader_row_count": strict_reader_row_count,
        "documented_transcript_row_count": DOCUMENTED_TRANSCRIPT_ROW_COUNT,
        "strict_reader_row_count_matches": strict_reader_row_count
        == DOCUMENTED_TRANSCRIPT_ROW_COUNT,
        "strict_reader_content_char_count": strict_reader_content_chars,
        "documented_transcript_content_char_count": DOCUMENTED_TRANSCRIPT_CONTENT_CHARS,
        "strict_reader_content_char_count_matches": strict_reader_content_chars
        == DOCUMENTED_TRANSCRIPT_CONTENT_CHARS,
        "passed": bool(
            all_match
            and session_views_unique
            and no_missing
            and strict_reader_invalid_count == 0
            and strict_reader_row_count == DOCUMENTED_TRANSCRIPT_ROW_COUNT
            and strict_reader_content_chars == DOCUMENTED_TRANSCRIPT_CONTENT_CHARS
        ),
    }


def _text_tuple_digest(values: Sequence[object]) -> str:
    return _sha256_fields(values, domain=b"trace-ace-session-text-v1\0")


def _hash_row_digests(values: Sequence[tuple[int, str]]) -> str:
    digest = hashlib.sha256(b"trace-ace-response-session-text-v1\0")
    for row_id, value in sorted(values):
        for field in (row_id, value):
            encoded = str(field).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _finite_dense_evidence(parquet: pq.ParquetFile) -> dict[str, object]:
    columns = list(dense_columns())
    non_finite = 0
    null_count = 0
    value_count = 0
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        for column in batch.columns:
            null_count += column.null_count
            if column.null_count:
                continue
            values = column.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            non_finite += int(np.size(values) - np.count_nonzero(np.isfinite(values)))
            value_count += int(np.size(values))
    return {
        "column_count": len(columns),
        "value_count": value_count,
        "null_count": null_count,
        "non_finite_count": non_finite,
        "all_finite": null_count == 0 and non_finite == 0,
    }


def _audit_response_views(
    root: Path, transcript_evidence: Mapping[str, object]
) -> dict[str, object]:
    cache_dir = root / "data" / "interim" / "cache"
    store_dir = root / "data" / "interim" / "response_views"
    response_path = cache_dir / "responses.parquet"
    sessions_path = cache_dir / "session_views.parquet"
    view_path = store_dir / "response_views.parquet"
    manifest_path = store_dir / "response_views_manifest.json"
    manifest = _load_json_object(manifest_path)
    sources = _safe_dict(manifest.get("sources"))

    parquet = pq.ParquetFile(view_path)
    schema_matches = parquet.schema_arrow == RESPONSE_VIEWS_SCHEMA
    view_sha = file_sha256(view_path)
    response_sha = file_sha256(response_path)
    session_sha = file_sha256(sessions_path)
    builder_sha = file_sha256(root / "src" / "trace_ace" / "feature_store.py")
    extractor_sha = file_sha256(root / "src" / "trace_ace" / "features.py")
    reader_sha = file_sha256(root / "src" / "trace_ace" / "io.py")
    expected_feature_config = asdict(
        FeatureConfig(context_char_budget=ModelConfig().context_char_budget)
    )
    expected_transcript_tree = str(
        transcript_evidence.get("extracted_feature_tree_sha256", "")
    )
    source_fingerprints_match = bool(
        sources.get("responses_sha256") == response_sha
        and sources.get("session_views_sha256") == session_sha
        and sources.get("feature_config") == expected_feature_config
        and sources.get("dense_feature_names") == list(DENSE_FEATURE_NAMES)
        and sources.get("transcript_tree_sha256") == expected_transcript_tree
    )
    code_fingerprints_match = bool(
        sources.get("builder_sha256") == builder_sha
        and sources.get("extractor_sha256") == extractor_sha
        and sources.get("reader_sha256") == reader_sha
    )

    cached = pq.read_table(
        response_path,
        columns=[
            "row_id",
            "response_id",
            "session_id",
            "learning_objective_id",
            "learning_objective",
            "is_correct",
        ],
    ).to_pandas()
    viewed = pq.read_table(
        view_path,
        columns=[
            "row_id",
            "response_id",
            "session_id",
            "learning_objective_id",
            "learning_objective",
            "is_correct",
        ],
    ).to_pandas()
    cached_identity = response_identity_sha256(cached)
    viewed_identity = response_identity_sha256(viewed)
    cached_objectives = _sha256_frame(
        cached,
        ("row_id", "learning_objective"),
        sort_by=("row_id",),
        domain=b"trace-ace-response-objective-text-v1\0",
    )
    viewed_objectives = _sha256_frame(
        viewed,
        ("row_id", "learning_objective"),
        sort_by=("row_id",),
        domain=b"trace-ace-response-objective-text-v1\0",
    )

    session_text: dict[str, str] = {}
    session_parquet = pq.ParquetFile(sessions_path)
    session_duplicate_count = 0
    for batch in session_parquet.iter_batches(
        batch_size=256, columns=["session_id", "full_text", "tutor_text", "student_text"]
    ):
        for row in batch.to_pylist():
            identity = str(row["session_id"])
            if identity in session_text:
                session_duplicate_count += 1
            if any(row[column] is None for column in ("full_text", "tutor_text", "student_text")):
                continue
            session_text[identity] = _text_tuple_digest(
                (row["full_text"], row["tutor_text"], row["student_text"])
            )

    expected_row_text: list[tuple[int, str]] = []
    observed_row_text: list[tuple[int, str]] = []
    text_mismatch_count = 0
    unknown_session_count = 0
    text_null_count = 0
    observed_row_ids: list[int] = []
    text_columns = [
        "row_id",
        "session_id",
        "full_text",
        "tutor_text",
        "student_text",
        "objective_context",
        "tutor_relevant_text",
        "student_relevant_text",
        "tutor_question_student_answer_windows",
        "student_answer_tutor_feedback_windows",
        "opening_evidence",
        "closing_evidence",
    ]
    for batch in parquet.iter_batches(batch_size=256, columns=text_columns):
        for row in batch.to_pylist():
            row_id = int(row["row_id"])
            observed_row_ids.append(row_id)
            text_null_count += sum(row[column] is None for column in text_columns[2:])
            expected_digest = session_text.get(str(row["session_id"]))
            if expected_digest is None:
                unknown_session_count += 1
                expected_digest = hashlib.sha256(b"").hexdigest()
            if any(row[column] is None for column in ("full_text", "tutor_text", "student_text")):
                observed_digest = hashlib.sha256(b"").hexdigest()
            else:
                observed_digest = _text_tuple_digest(
                    (row["full_text"], row["tutor_text"], row["student_text"])
                )
            expected_row_text.append((row_id, expected_digest))
            observed_row_text.append((row_id, observed_digest))
            text_mismatch_count += expected_digest != observed_digest

    expected_text_hash = _hash_row_digests(expected_row_text)
    observed_text_hash = _hash_row_digests(observed_row_text)
    row_ids_contiguous = np.array_equal(
        np.sort(np.asarray(observed_row_ids, dtype=np.int64)),
        np.arange(len(observed_row_ids), dtype=np.int64),
    )
    dense_evidence = _finite_dense_evidence(parquet)
    output_manifest_matches = bool(
        manifest.get("output_sha256") == view_sha
        and _safe_int(manifest.get("response_count")) == parquet.metadata.num_rows
        and _safe_int(manifest.get("session_count")) == len(session_text)
        and manifest.get("schema") == str(RESPONSE_VIEWS_SCHEMA)
        and schema_matches
    )
    version_matches = manifest.get("version") == FEATURE_STORE_VERSION
    parity_passed = bool(
        len(cached) == len(viewed) == parquet.metadata.num_rows
        and row_ids_contiguous
        and cached_identity == viewed_identity
        and cached_objectives == viewed_objectives
        and session_duplicate_count == 0
        and unknown_session_count == 0
        and text_null_count == 0
        and text_mismatch_count == 0
        and expected_text_hash == observed_text_hash
    )
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "version_matches": version_matches,
        "output_sha256": view_sha,
        "row_count": parquet.metadata.num_rows,
        "schema_sha256": _sha256_schema(parquet.schema_arrow),
        "expected_schema_sha256": _sha256_schema(RESPONSE_VIEWS_SCHEMA),
        "schema_matches": schema_matches,
        "output_manifest_matches": output_manifest_matches,
        "cache_responses_sha256": response_sha,
        "cache_session_views_sha256": session_sha,
        "source_transcript_tree_sha256": expected_transcript_tree,
        "source_fingerprints_match": source_fingerprints_match,
        "builder_sha256": builder_sha,
        "extractor_sha256": extractor_sha,
        "reader_sha256": reader_sha,
        "code_fingerprints_match": code_fingerprints_match,
        "cache_identity_sha256": cached_identity,
        "response_view_identity_sha256": viewed_identity,
        "identity_matches": cached_identity == viewed_identity,
        "cache_objective_text_sha256": cached_objectives,
        "response_view_objective_text_sha256": viewed_objectives,
        "objective_text_matches": cached_objectives == viewed_objectives,
        "session_text_expected_sha256": expected_text_hash,
        "session_text_observed_sha256": observed_text_hash,
        "session_text_matches": expected_text_hash == observed_text_hash,
        "session_text_mismatch_count": text_mismatch_count,
        "unknown_session_count": unknown_session_count,
        "session_duplicate_count": session_duplicate_count,
        "text_null_count": text_null_count,
        "row_ids_contiguous": row_ids_contiguous,
        "dense": dense_evidence,
        "parity_passed": parity_passed,
        "passed": bool(
            version_matches
            and output_manifest_matches
            and source_fingerprints_match
            and code_fingerprints_match
            and parity_passed
            and dense_evidence["all_finite"]
        ),
    }


def _expected_model_rows(view_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    identity = pq.read_table(
        view_path,
        columns=[
            "row_id",
            "response_id",
            "session_id",
            "learning_objective_id",
            "is_correct",
            *dense_columns(),
        ],
    ).to_pandas().sort_values("row_id", kind="stable")
    row_ids = identity["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(len(identity), dtype=np.int64)):
        raise ValueError("non-contiguous model rows")
    targets = identity["is_correct"].to_numpy(dtype=np.int8)
    dense = identity.loc[:, dense_columns()].to_numpy(dtype=np.float32)
    return row_ids, targets, dense, response_identity_sha256(identity)


def _audit_sparse_store(root: Path) -> dict[str, object]:
    store = root / "data" / "processed" / "sparse_store"
    if not store.exists():
        return {"present": False, "optional": True, "passed": True}
    if not (store / "manifest.json").is_file():
        return {"present": True, "optional": True, "hash_validation_passed": False, "passed": False}
    try:
        manifest = validate_sparse_store(store)
    except Exception:
        return {"present": True, "optional": True, "hash_validation_passed": False, "passed": False}

    view_path = root / "data" / "interim" / "response_views" / "response_views.parquet"
    _, expected_targets, expected_dense, identity_sha = _expected_model_rows(view_path)
    loaded_targets, loaded_dense = load_dense_by_row(store)
    source = _safe_dict(manifest.get("source"))
    code = _safe_dict(source.get("code"))
    source_sha = file_sha256(view_path)
    source_fingerprint_matches = bool(
        source.get("source_sha256") == source_sha
        and source.get("response_identity_sha256") == identity_sha
        and source.get("config") == ModelConfig().to_dict()
        and source.get("dense_columns") == list(dense_columns())
        and _safe_int(source.get("batch_size"), 0) > 0
    )
    expected_builder = file_sha256(root / "src" / "trace_ace" / "sparse_store.py")
    expected_transform = file_sha256(root / "src" / "trace_ace" / "sparse.py")
    code_fingerprints_match = bool(
        code.get("builder_sha256") == expected_builder
        and code.get("transform_sha256") == expected_transform
    )

    sparse_values_finite = True
    part_row_count = 0
    part_count = 0
    for part in iter_sparse_parts(store):
        part_count += 1
        part_row_count += len(part.row_ids)
        sparse_values_finite &= bool(
            np.isfinite(part.full.data).all()
            and np.isfinite(part.role.data).all()
            and np.isfinite(part.dense).all()
        )
    targets_match = np.array_equal(loaded_targets, expected_targets)
    dense_match = np.array_equal(loaded_dense, expected_dense)
    dense_finite = bool(np.isfinite(loaded_dense).all())
    response_count = _safe_int(manifest.get("response_count"))
    manifest_part_count = _safe_int(manifest.get("part_count"))
    coverage_matches = bool(
        response_count == len(expected_targets) == len(loaded_targets) == part_row_count
        and manifest_part_count == part_count
    )
    return {
        "present": True,
        "optional": True,
        "manifest_sha256": file_sha256(store / "manifest.json"),
        "version_matches": manifest.get("version") == SPARSE_STORE_VERSION,
        "hash_validation_passed": True,
        "source_sha256": source_sha,
        "source_identity_sha256": identity_sha,
        "source_fingerprint_matches": source_fingerprint_matches,
        "builder_sha256": expected_builder,
        "transform_sha256": expected_transform,
        "code_fingerprints_match": code_fingerprints_match,
        "response_count": response_count,
        "part_count": part_count,
        "coverage_matches": coverage_matches,
        "targets_match": targets_match,
        "dense_matches": dense_match,
        "dense_all_finite": dense_finite,
        "sparse_values_all_finite": sparse_values_finite,
        "passed": bool(
            manifest.get("version") == SPARSE_STORE_VERSION
            and source_fingerprint_matches
            and code_fingerprints_match
            and coverage_matches
            and targets_match
            and dense_match
            and dense_finite
            and sparse_values_finite
        ),
    }


def _array_all_finite(values: np.ndarray, *, batch_size: int = 4096) -> bool:
    return all(
        np.isfinite(values[start : start + batch_size]).all()
        for start in range(0, len(values), batch_size)
    )


def _audit_semantic_store(root: Path) -> dict[str, object]:
    store = root / "data" / "processed" / "semantic_store"
    if not store.exists():
        return {"present": False, "optional": True, "passed": True}
    if not (store / "manifest.json").is_file():
        return {"present": True, "optional": True, "hash_validation_passed": False, "passed": False}
    try:
        manifest = load_semantic_manifest(store)
        arrays = load_semantic_arrays(store)
    except Exception:
        return {"present": True, "optional": True, "hash_validation_passed": False, "passed": False}

    view_path = root / "data" / "interim" / "response_views" / "response_views.parquet"
    _, expected_targets, expected_dense, identity_sha = _expected_model_rows(view_path)
    source_sha = file_sha256(view_path)
    source = _safe_dict(manifest.get("source"))
    code = _safe_dict(source.get("code"))
    asset = root / "assets" / "bge-small-en-v1.5"
    asset_hash = asset_tree_sha256(asset)
    model_hash = file_sha256(asset / "model.safetensors")
    source_fingerprint_matches = bool(
        source.get("source_sha256") == source_sha
        and source.get("response_identity_sha256") == identity_sha
        and source.get("model_sha256") == model_hash
        and source.get("asset_tree_sha256") == asset_hash
        and source.get("config") == ModelConfig().to_dict()
        and source.get("device_independent_format") == "normalized-float32"
    )
    expected_builder = file_sha256(root / "src" / "trace_ace" / "semantic_store.py")
    expected_encoder = file_sha256(root / "src" / "trace_ace" / "semantic.py")
    code_fingerprints_match = bool(
        code.get("builder_sha256") == expected_builder
        and code.get("encoder_sha256") == expected_encoder
    )
    row_count = _safe_int(manifest.get("response_count"))
    objective_count = _safe_int(manifest.get("objective_count"))
    dimension = _safe_int(manifest.get("embedding_dimension"))
    indices = arrays.objective_index_by_row
    objective_rows = pq.read_table(
        view_path, columns=["row_id", "learning_objective_id"]
    ).to_pandas().sort_values("row_id", kind="stable")
    expected_objective_ids = np.asarray(
        sorted(objective_rows["learning_objective_id"].astype(str).unique()),
        dtype=np.str_,
    )
    expected_objective_lookup = {
        value: index for index, value in enumerate(expected_objective_ids)
    }
    expected_indices = (
        objective_rows["learning_objective_id"]
        .astype(str)
        .map(expected_objective_lookup)
        .to_numpy(dtype=np.int32)
    )
    objective_ids_match = np.array_equal(arrays.objective_ids, expected_objective_ids)
    objective_index_mapping_matches = np.array_equal(indices, expected_indices)
    expected_objective_identity_sha = _sha256_fields(
        expected_objective_ids,
        domain=b"trace-ace-semantic-objective-identities-v1\0",
    )
    stored_objective_identity_sha = _sha256_fields(
        arrays.objective_ids,
        domain=b"trace-ace-semantic-objective-identities-v1\0",
    )
    indices_valid = bool(
        len(indices) == row_count
        and objective_count > 0
        and np.all(indices >= 0)
        and np.all(indices < objective_count)
        and len(np.unique(indices)) == objective_count
    )
    shapes_match = bool(
        arrays.context_embeddings.shape == (row_count, dimension)
        and arrays.objective_embeddings.shape == (objective_count, dimension)
        and arrays.dense.shape == expected_dense.shape
        and len(arrays.objective_ids) == objective_count
        and dimension == BGE_DIMENSION
    )
    identities_unique = len(np.unique(arrays.objective_ids)) == len(arrays.objective_ids)
    targets_match = np.array_equal(arrays.targets, expected_targets)
    dense_match = np.array_equal(arrays.dense, expected_dense)
    context_finite = _array_all_finite(arrays.context_embeddings)
    objective_finite = _array_all_finite(arrays.objective_embeddings)
    dense_finite = bool(np.isfinite(arrays.dense).all())
    return {
        "present": True,
        "optional": True,
        "manifest_sha256": file_sha256(store / "manifest.json"),
        "version_matches": manifest.get("version") == SEMANTIC_STORE_VERSION,
        "hash_validation_passed": True,
        "source_sha256": source_sha,
        "source_identity_sha256": identity_sha,
        "asset_tree_sha256": asset_hash,
        "model_sha256": model_hash,
        "source_fingerprint_matches": source_fingerprint_matches,
        "builder_sha256": expected_builder,
        "encoder_sha256": expected_encoder,
        "code_fingerprints_match": code_fingerprints_match,
        "response_count": row_count,
        "objective_count": objective_count,
        "embedding_dimension": dimension,
        "shapes_match": shapes_match,
        "objective_indices_valid": indices_valid,
        "objective_identities_unique": identities_unique,
        "expected_objective_identity_sha256": expected_objective_identity_sha,
        "stored_objective_identity_sha256": stored_objective_identity_sha,
        "objective_identities_match": objective_ids_match,
        "objective_index_mapping_matches": objective_index_mapping_matches,
        "targets_match": targets_match,
        "dense_matches": dense_match,
        "context_embeddings_all_finite": context_finite,
        "objective_embeddings_all_finite": objective_finite,
        "dense_all_finite": dense_finite,
        "passed": bool(
            manifest.get("version") == SEMANTIC_STORE_VERSION
            and source_fingerprint_matches
            and code_fingerprints_match
            and shapes_match
            and indices_valid
            and identities_unique
            and objective_ids_match
            and objective_index_mapping_matches
            and targets_match
            and dense_match
            and context_finite
            and objective_finite
            and dense_finite
        ),
    }


def audit_project(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Run every read-only gate and return non-sensitive JSON evidence."""

    root = root.resolve()
    checks: dict[str, dict[str, object]] = {}
    checks["documented_sources"] = _guard(lambda: _audit_documented_sources(root))
    checks["source_raw_csv_parity"] = _guard(
        lambda: _audit_source_raw_csv_parity(root)
    )
    checks["transcript_archive_parity"] = _guard(
        lambda: _audit_transcript_archive(
            root / "data" / "source" / str(SOURCE_SPECS["transcript_archive"]["filename"]),
            root / "data" / "raw" / "train_transcripts",
        )
    )
    transcript_evidence = checks["transcript_archive_parity"]
    checks["cache_manifest"] = _guard(
        lambda: _audit_cache(root, transcript_evidence)
    )
    checks["source_cache_response_parity"] = _guard(
        lambda: _audit_response_parity(root)
    )
    checks["session_identity_coverage"] = _guard(
        lambda: _audit_session_coverage(root)
    )
    checks["response_views"] = _guard(
        lambda: _audit_response_views(root, transcript_evidence)
    )
    checks["sparse_store"] = _guard(lambda: _audit_sparse_store(root))
    checks["semantic_store"] = _guard(lambda: _audit_semantic_store(root))
    passed = all(bool(check.get("passed", False)) for check in checks.values())
    return {"audit_version": AUDIT_VERSION, "passed": passed, "checks": checks}


def main() -> int:
    evidence = audit_project()
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
