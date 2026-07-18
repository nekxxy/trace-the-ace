"""Stream per-response, independently computable modeling views to Parquet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trace_ace.features import (
    DENSE_FEATURE_NAMES,
    FeatureConfig,
    extract_objective_features,
)
from trace_ace.io import read_transcript
from trace_ace.provenance import file_sha256


FEATURE_STORE_VERSION = "trace-ace-response-views-v2"
TEXT_VIEW_COLUMNS = (
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
)


def _schema() -> pa.Schema:
    fields = [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("learning_objective_id", pa.string(), nullable=False),
        pa.field("learning_objective", pa.string(), nullable=False),
        pa.field("is_correct", pa.int8(), nullable=False),
    ]
    fields.extend(pa.field(name, pa.string(), nullable=False) for name in TEXT_VIEW_COLUMNS)
    fields.extend(
        pa.field(f"dense__{name}", pa.float32(), nullable=False)
        for name in DENSE_FEATURE_NAMES
    )
    return pa.schema(fields)


RESPONSE_VIEWS_SCHEMA = _schema()


@dataclass(frozen=True)
class FeatureStoreResult:
    status: str
    path: Path
    manifest_path: Path
    response_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transcript_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(root.glob("*.csv"), key=lambda path: path.name)
    if not files:
        raise ValueError("transcript directory contains no CSV files")
    for path in files:
        encoded = path.name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_transcript_path(root: Path, session_id: str) -> Path:
    if not session_id or Path(session_id).name != session_id:
        raise ValueError("session ID is not a safe transcript basename")
    candidate = (root / f"{session_id}.csv").resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("transcript path escapes its data directory") from error
    return candidate


def _temporary_path(directory: Path, name: str) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
    os.close(descriptor)
    return Path(raw)


def _manifest_matches(
    manifest_path: Path,
    output_path: Path,
    expected_sources: dict[str, object],
) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return bool(
        manifest.get("version") == FEATURE_STORE_VERSION
        and manifest.get("sources") == expected_sources
        and output_path.is_file()
        and manifest.get("output_sha256") == _sha256(output_path)
    )


def build_response_views(
    *,
    responses_path: str | Path,
    session_views_path: str | Path,
    transcripts_dir: str | Path,
    output_dir: str | Path,
    feature_config: FeatureConfig | None = None,
    force: bool = False,
    batch_size: int = 32,
    progress: Callable[[str], None] | None = None,
) -> FeatureStoreResult:
    """Build a deterministic response-view cache with bounded memory.

    Each response is computed only from its own objective and session
    transcript. No statistic is learned from other response rows.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    cfg = feature_config or FeatureConfig()
    responses_file = Path(responses_path).resolve()
    sessions_file = Path(session_views_path).resolve()
    transcript_root = Path(transcripts_dir).resolve()
    destination = Path(output_dir).resolve()
    if not responses_file.is_file() or not sessions_file.is_file():
        raise FileNotFoundError("responses and session-view caches are required")
    if not transcript_root.is_dir():
        raise FileNotFoundError(f"transcript directory not found: {transcript_root}")
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "response_views.parquet"
    manifest_path = destination / "response_views_manifest.json"

    source_fingerprint: dict[str, object] = {
        "responses_sha256": _sha256(responses_file),
        "session_views_sha256": _sha256(sessions_file),
        "feature_config": asdict(cfg),
        "dense_feature_names": list(DENSE_FEATURE_NAMES),
        "transcript_tree_sha256": _transcript_tree_sha256(transcript_root),
        "builder_sha256": file_sha256(Path(__file__)),
        "extractor_sha256": file_sha256(Path(__file__).with_name("features.py")),
        "reader_sha256": file_sha256(Path(__file__).with_name("io.py")),
    }
    if not force and _manifest_matches(manifest_path, output_path, source_fingerprint):
        row_count = pq.ParquetFile(output_path).metadata.num_rows
        if progress:
            progress("response views: unchanged inputs; existing cache is valid")
        return FeatureStoreResult("reused", output_path, manifest_path, row_count)

    responses = pq.read_table(responses_file).to_pandas()
    by_session = {
        str(session_id): group.sort_values("row_id", kind="stable")
        for session_id, group in responses.groupby("session_id", sort=False)
    }
    expected_sessions = set(by_session)
    observed_sessions: set[str] = set()
    expected_rows = len(responses)

    temp_output = _temporary_path(destination, output_path.name)
    temp_manifest = _temporary_path(destination, manifest_path.name)
    writer = pq.ParquetWriter(
        temp_output,
        RESPONSE_VIEWS_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    buffer: list[dict[str, object]] = []
    written = 0
    session_count = 0
    try:
        session_parquet = pq.ParquetFile(sessions_file)
        for record_batch in session_parquet.iter_batches(batch_size=batch_size):
            for session_view in record_batch.to_pylist():
                session_id = str(session_view["session_id"])
                if session_id not in by_session:
                    raise ValueError(f"session view has no response rows: {session_id}")
                if session_id in observed_sessions:
                    raise ValueError(f"duplicate session view: {session_id}")
                observed_sessions.add(session_id)
                transcript_path = _safe_transcript_path(transcript_root, session_id)
                if not transcript_path.is_file():
                    raise FileNotFoundError(f"missing transcript: {transcript_path.name}")
                transcript = read_transcript(
                    transcript_path, expected_session_id=session_id
                )

                for response in by_session[session_id].itertuples(index=False):
                    extracted = extract_objective_features(
                        transcript,
                        response.learning_objective,
                        config=cfg,
                    )
                    row: dict[str, object] = {
                        "row_id": int(response.row_id),
                        "response_id": str(response.response_id),
                        "session_id": session_id,
                        "learning_objective_id": str(response.learning_objective_id),
                        "learning_objective": str(response.learning_objective),
                        "is_correct": int(response.is_correct),
                        "full_text": str(session_view["full_text"]),
                        "tutor_text": str(session_view["tutor_text"]),
                        "student_text": str(session_view["student_text"]),
                        "objective_context": extracted.objective_context,
                        "tutor_relevant_text": extracted.tutor_relevant_text,
                        "student_relevant_text": extracted.student_relevant_text,
                        "tutor_question_student_answer_windows": (
                            extracted.tutor_question_student_answer_windows
                        ),
                        "student_answer_tutor_feedback_windows": (
                            extracted.student_answer_tutor_feedback_windows
                        ),
                        "opening_evidence": extracted.opening_evidence,
                        "closing_evidence": extracted.closing_evidence,
                    }
                    row.update(
                        {
                            f"dense__{name}": float(value)
                            for name, value in zip(
                                DENSE_FEATURE_NAMES, extracted.dense, strict=True
                            )
                        }
                    )
                    buffer.append(row)
                    if len(buffer) >= batch_size:
                        writer.write_table(
                            pa.Table.from_pylist(buffer, schema=RESPONSE_VIEWS_SCHEMA)
                        )
                        written += len(buffer)
                        buffer.clear()
                session_count += 1
                if progress and (session_count % 1000 == 0):
                    progress(f"response views: processed {session_count} sessions")

        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=RESPONSE_VIEWS_SCHEMA))
            written += len(buffer)
            buffer.clear()
    finally:
        writer.close()
        if sys.exc_info()[0] is not None:
            temp_output.unlink(missing_ok=True)
            temp_manifest.unlink(missing_ok=True)

    try:
        if observed_sessions != expected_sessions:
            missing = len(expected_sessions - observed_sessions)
            raise ValueError(
                f"session views did not cover all response sessions; missing={missing}"
            )
        if written != expected_rows:
            raise ValueError(
                f"response-view row count mismatch: {written} != {expected_rows}"
            )

        output_sha256 = _sha256(temp_output)
        manifest = {
            "version": FEATURE_STORE_VERSION,
            "sources": source_fingerprint,
            "response_count": written,
            "session_count": session_count,
            "schema": str(RESPONSE_VIEWS_SCHEMA),
            "output_sha256": output_sha256,
        }
        temp_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp_output, output_path)
        os.replace(temp_manifest, manifest_path)
    except BaseException:
        temp_output.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
        raise
    if progress:
        progress(f"response views: completed {written} responses")
    return FeatureStoreResult("built", output_path, manifest_path, written)


def dense_columns() -> tuple[str, ...]:
    return tuple(f"dense__{name}" for name in DENSE_FEATURE_NAMES)
