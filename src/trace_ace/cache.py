"""Deterministic, memory-bounded training cache construction.

Only one transcript is loaded at a time.  Aggregated session rows are buffered in
small batches and written through :class:`pyarrow.parquet.ParquetWriter`, so the
full transcript corpus is never resident in memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trace_ace.io import read_transcript


CACHE_VERSION = "trace-ace-cache-v2"

FEATURE_COLUMNS = [
    "response_id",
    "session_id",
    "learning_objective_id",
    "learning_objective",
]
LABEL_COLUMNS = ["response_id", "is_correct"]
TRANSCRIPT_COLUMNS = ["session_id", "utterance_id", "role", "content", "timestamp"]
ALLOWED_ROLES = ("tutor", "student", "background")

OPENING_UTTERANCES = 8
CLOSING_UTTERANCES_PER_ROLE = 3
SHORT_STUDENT_MAX_WORDS = 5

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
UNCLEAR_RE = re.compile(
    r"\b(?:idk|unsure|confused|unclear)\b|"
    r"\bi\s+(?:do\s*not|don't|dont)\s+know\b|"
    r"\b(?:do\s*not|don't|dont)\s+understand\b|"
    r"\bnot\s+sure\b|\bno\s+idea\b",
    flags=re.IGNORECASE,
)
FEEDBACK_RE = re.compile(
    r"\b(?:correct|exactly|excellent|good|great|nice|yes)\b|"
    r"\bwell\s+done\b|\bthat(?:'s|\s+is)\s+right\b|\byou\s+got\s+it\b",
    flags=re.IGNORECASE,
)
CORRECTION_RE = re.compile(
    r"\b(?:incorrect|mistake|instead|actually|careful)\b|"
    r"\bnot\s+quite\b|\btry\s+again\b|\blet(?:'s|\s+us)\s+correct\b",
    flags=re.IGNORECASE,
)


RESPONSES_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("learning_objective_id", pa.string(), nullable=False),
        pa.field("learning_objective", pa.string(), nullable=False),
        pa.field("is_correct", pa.int8(), nullable=False),
    ]
)

SESSION_VIEWS_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("full_text", pa.string(), nullable=False),
        pa.field("tutor_text", pa.string(), nullable=False),
        pa.field("student_text", pa.string(), nullable=False),
        pa.field("background_text", pa.string(), nullable=False),
        pa.field("opening_text", pa.string(), nullable=False),
        pa.field("closing_tutor_text", pa.string(), nullable=False),
        pa.field("closing_student_text", pa.string(), nullable=False),
        pa.field("utterance_count", pa.int64(), nullable=False),
        pa.field("tutor_utterance_count", pa.int64(), nullable=False),
        pa.field("student_utterance_count", pa.int64(), nullable=False),
        pa.field("background_utterance_count", pa.int64(), nullable=False),
        pa.field("tutor_utterance_ratio", pa.float64(), nullable=False),
        pa.field("student_utterance_ratio", pa.float64(), nullable=False),
        pa.field("background_utterance_ratio", pa.float64(), nullable=False),
        pa.field("char_count", pa.int64(), nullable=False),
        pa.field("tutor_char_count", pa.int64(), nullable=False),
        pa.field("student_char_count", pa.int64(), nullable=False),
        pa.field("background_char_count", pa.int64(), nullable=False),
        pa.field("tutor_char_ratio", pa.float64(), nullable=False),
        pa.field("student_char_ratio", pa.float64(), nullable=False),
        pa.field("background_char_ratio", pa.float64(), nullable=False),
        pa.field("word_count", pa.int64(), nullable=False),
        pa.field("tutor_word_count", pa.int64(), nullable=False),
        pa.field("student_word_count", pa.int64(), nullable=False),
        pa.field("background_word_count", pa.int64(), nullable=False),
        pa.field("tutor_word_ratio", pa.float64(), nullable=False),
        pa.field("student_word_ratio", pa.float64(), nullable=False),
        pa.field("background_word_ratio", pa.float64(), nullable=False),
        pa.field("duration_seconds", pa.float64(), nullable=False),
        pa.field("unclear_student_utterance_count", pa.int64(), nullable=False),
        pa.field("unclear_student_utterance_ratio", pa.float64(), nullable=False),
        pa.field("question_utterance_count", pa.int64(), nullable=False),
        pa.field("question_utterance_ratio", pa.float64(), nullable=False),
        pa.field("role_transition_count", pa.int64(), nullable=False),
        pa.field("role_transition_ratio", pa.float64(), nullable=False),
        pa.field("short_student_utterance_count", pa.int64(), nullable=False),
        pa.field("short_student_utterance_ratio", pa.float64(), nullable=False),
        pa.field("feedback_proxy_count", pa.int64(), nullable=False),
        pa.field("feedback_proxy_ratio", pa.float64(), nullable=False),
        pa.field("correction_proxy_count", pa.int64(), nullable=False),
        pa.field("correction_proxy_ratio", pa.float64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class CacheSources:
    """Resolved source files for one cache build."""

    raw_dir: Path
    features_path: Path
    labels_path: Path
    transcripts_dir: Path
    transcript_files: tuple[Path, ...]
    transcript_archive: Path | None


@dataclass(frozen=True)
class CacheBuildResult:
    """Summary returned by :func:`build_cache`."""

    status: str
    responses_path: Path
    session_views_path: Path
    manifest_path: Path
    response_count: int
    session_count: int


def _one_candidate(candidates: Iterable[Path], description: str) -> Path:
    found = sorted((path.resolve() for path in candidates), key=lambda path: path.name.lower())
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {description}; found {len(found)}")
    return found[0]


def discover_sources(raw_dir: str | Path) -> CacheSources:
    """Autodiscover the official feature, label, and transcript inputs."""

    root = Path(raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")

    children = list(root.iterdir())
    supported_table_suffixes = {".csv", ".parquet"}
    features_path = _one_candidate(
        (
            path
            for path in children
            if path.is_file()
            and path.suffix.lower() in supported_table_suffixes
            and path.name.lower().startswith("train_features")
        ),
        "train_features* table",
    )
    labels_path = _one_candidate(
        (
            path
            for path in children
            if path.is_file()
            and path.suffix.lower() in supported_table_suffixes
            and path.name.lower().startswith("train_labels")
        ),
        "train_labels* table",
    )
    transcripts_dir = _one_candidate(
        (
            path
            for path in children
            if path.is_dir()
            and "train" in path.name.lower()
            and "transcript" in path.name.lower()
        ),
        "train transcript directory",
    )

    transcript_files = tuple(
        sorted(transcripts_dir.glob("*.csv"), key=lambda path: path.name)
    )
    if not transcript_files:
        raise ValueError("The train transcript directory contains no CSV files")
    stems = [path.stem for path in transcript_files]
    if len(stems) != len(set(stems)):
        raise ValueError("Transcript filenames do not map one-to-one to session IDs")

    archives = sorted(
        (
            path.resolve()
            for path in children
            if path.is_file()
            and path.suffix.lower() == ".zip"
            and "train" in path.name.lower()
            and "transcript" in path.name.lower()
        ),
        key=lambda path: path.name.lower(),
    )
    if len(archives) > 1:
        raise ValueError(f"Expected at most one transcript archive; found {len(archives)}")

    return CacheSources(
        raw_dir=root,
        features_path=features_path,
        labels_path=labels_path,
        transcripts_dir=transcripts_dir,
        transcript_files=transcript_files,
        transcript_archive=archives[0] if archives else None,
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).astype("string")
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _validate_columns(frame: pd.DataFrame, expected: list[str], description: str) -> None:
    actual = list(frame.columns)
    if actual != expected:
        raise ValueError(
            f"Invalid {description} schema: expected columns {expected}, found {actual}"
        )


def _require_nonempty(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    if frame.empty:
        raise ValueError(f"{description} is empty")
    for column in columns:
        values = frame[column].astype("string")
        if values.isna().any() or values.str.len().eq(0).any():
            raise ValueError(f"{description} has missing values in {column}")


def _load_and_validate_response_tables(
    sources: CacheSources,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    features = _read_table(sources.features_path)
    labels = _read_table(sources.labels_path)
    _validate_columns(features, FEATURE_COLUMNS, "train features")
    _validate_columns(labels, LABEL_COLUMNS, "train labels")
    _require_nonempty(features, FEATURE_COLUMNS, "train features")
    _require_nonempty(labels, LABEL_COLUMNS, "train labels")

    if features["response_id"].duplicated().any():
        raise ValueError("train features response_id values are not one-to-one")
    if labels["response_id"].duplicated().any():
        raise ValueError("train labels response_id values are not one-to-one")

    feature_ids = set(features["response_id"].tolist())
    label_ids = set(labels["response_id"].tolist())
    if feature_ids != label_ids:
        raise ValueError("Feature and label response IDs do not match one-to-one")

    numeric_labels = pd.to_numeric(labels["is_correct"], errors="raise")
    if not numeric_labels.isin([0, 1]).all():
        raise ValueError("is_correct must contain only binary 0/1 labels")
    labels = labels.copy()
    labels["is_correct"] = numeric_labels.astype("int8")

    merged = features.merge(
        labels,
        on="response_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    merged = merged.sort_values("response_id", kind="mergesort").reset_index(drop=True)
    merged.insert(0, "row_id", range(len(merged)))

    transcript_by_session = {path.stem: path for path in sources.transcript_files}
    feature_sessions = set(merged["session_id"].tolist())
    transcript_sessions = set(transcript_by_session)
    if feature_sessions != transcript_sessions:
        raise ValueError("Feature sessions and transcript files do not match one-to-one")

    return merged, transcript_by_session


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transcript_tree_sha256(sources: CacheSources) -> str:
    digest = hashlib.sha256()
    digest.update(b"trace-ace-transcript-tree-v1\0")
    for path in sources.transcript_files:
        relative = path.relative_to(sources.transcripts_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _source_fingerprint(sources: CacheSources) -> dict[str, object]:
    fingerprint: dict[str, object] = {
        "code": {
            "builder_sha256": _sha256_file(Path(__file__)),
            "reader_sha256": _sha256_file(Path(__file__).with_name("io.py")),
        },
        "features": {
            "path": sources.features_path.relative_to(sources.raw_dir).as_posix(),
            "sha256": _sha256_file(sources.features_path),
        },
        "labels": {
            "path": sources.labels_path.relative_to(sources.raw_dir).as_posix(),
            "sha256": _sha256_file(sources.labels_path),
        },
        "transcripts": {
            "path": sources.transcripts_dir.relative_to(sources.raw_dir).as_posix(),
            "file_count": len(sources.transcript_files),
            "sha256": _transcript_tree_sha256(sources),
        },
    }
    if sources.transcript_archive is not None:
        fingerprint["transcript_archive"] = {
            "path": sources.transcript_archive.relative_to(sources.raw_dir).as_posix(),
            "sha256": _sha256_file(sources.transcript_archive),
        }
    return fingerprint


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _timestamp_seconds(value: object) -> float:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) == 3:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        except ValueError:
            pass
        else:
            if hours >= 0 and 0 <= minutes < 60 and 0 <= seconds < 60:
                return float(hours * 3600 + minutes * 60 + seconds)
    try:
        timestamp = pd.to_datetime(text, utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Transcript contains an invalid timestamp") from error
    return float(timestamp.value / 1_000_000_000)


def _read_transcript(path: Path, expected_session_id: str) -> pd.DataFrame:
    return read_transcript(path, expected_session_id=expected_session_id)


def _session_view(path: Path, session_id: str) -> dict[str, object]:
    frame = _read_transcript(path, session_id)
    roles = frame["role"].astype(str).tolist()
    contents = frame["content"].astype(str).tolist()
    role_prefixed = [f"{role.upper()}: {content}" for role, content in zip(roles, contents)]

    by_role: dict[str, list[str]] = {
        role: [content for row_role, content in zip(roles, contents) if row_role == role]
        for role in ALLOWED_ROLES
    }
    role_utterance_counts = {role: len(by_role[role]) for role in ALLOWED_ROLES}
    role_char_counts = {
        role: sum(len(content) for content in by_role[role]) for role in ALLOWED_ROLES
    }
    role_word_counts = {
        role: sum(len(WORD_RE.findall(content)) for content in by_role[role])
        for role in ALLOWED_ROLES
    }

    utterance_count = len(contents)
    char_count = sum(role_char_counts.values())
    word_count = sum(role_word_counts.values())
    timestamp_values = [_timestamp_seconds(value) for value in frame["timestamp"].tolist()]
    duration_seconds = max(timestamp_values) - min(timestamp_values)

    student_contents = by_role["student"]
    tutor_contents = by_role["tutor"]
    unclear_count = sum(bool(UNCLEAR_RE.search(content)) for content in student_contents)
    question_count = sum("?" in content for content in contents)
    transition_count = sum(left != right for left, right in zip(roles, roles[1:]))
    short_student_count = sum(
        len(WORD_RE.findall(content)) <= SHORT_STUDENT_MAX_WORDS
        for content in student_contents
    )
    feedback_count = sum(bool(FEEDBACK_RE.search(content)) for content in tutor_contents)
    correction_count = sum(bool(CORRECTION_RE.search(content)) for content in tutor_contents)

    return {
        "session_id": session_id,
        "full_text": "\n".join(role_prefixed),
        "tutor_text": "\n".join(by_role["tutor"]),
        "student_text": "\n".join(by_role["student"]),
        "background_text": "\n".join(by_role["background"]),
        "opening_text": "\n".join(role_prefixed[:OPENING_UTTERANCES]),
        "closing_tutor_text": "\n".join(
            by_role["tutor"][-CLOSING_UTTERANCES_PER_ROLE:]
        ),
        "closing_student_text": "\n".join(
            by_role["student"][-CLOSING_UTTERANCES_PER_ROLE:]
        ),
        "utterance_count": utterance_count,
        "tutor_utterance_count": role_utterance_counts["tutor"],
        "student_utterance_count": role_utterance_counts["student"],
        "background_utterance_count": role_utterance_counts["background"],
        "tutor_utterance_ratio": _safe_ratio(
            role_utterance_counts["tutor"], utterance_count
        ),
        "student_utterance_ratio": _safe_ratio(
            role_utterance_counts["student"], utterance_count
        ),
        "background_utterance_ratio": _safe_ratio(
            role_utterance_counts["background"], utterance_count
        ),
        "char_count": char_count,
        "tutor_char_count": role_char_counts["tutor"],
        "student_char_count": role_char_counts["student"],
        "background_char_count": role_char_counts["background"],
        "tutor_char_ratio": _safe_ratio(role_char_counts["tutor"], char_count),
        "student_char_ratio": _safe_ratio(role_char_counts["student"], char_count),
        "background_char_ratio": _safe_ratio(role_char_counts["background"], char_count),
        "word_count": word_count,
        "tutor_word_count": role_word_counts["tutor"],
        "student_word_count": role_word_counts["student"],
        "background_word_count": role_word_counts["background"],
        "tutor_word_ratio": _safe_ratio(role_word_counts["tutor"], word_count),
        "student_word_ratio": _safe_ratio(role_word_counts["student"], word_count),
        "background_word_ratio": _safe_ratio(role_word_counts["background"], word_count),
        "duration_seconds": float(duration_seconds),
        "unclear_student_utterance_count": unclear_count,
        "unclear_student_utterance_ratio": _safe_ratio(
            unclear_count, role_utterance_counts["student"]
        ),
        "question_utterance_count": question_count,
        "question_utterance_ratio": _safe_ratio(question_count, utterance_count),
        "role_transition_count": transition_count,
        "role_transition_ratio": _safe_ratio(transition_count, utterance_count - 1),
        "short_student_utterance_count": short_student_count,
        "short_student_utterance_ratio": _safe_ratio(
            short_student_count, role_utterance_counts["student"]
        ),
        "feedback_proxy_count": feedback_count,
        "feedback_proxy_ratio": _safe_ratio(
            feedback_count, role_utterance_counts["tutor"]
        ),
        "correction_proxy_count": correction_count,
        "correction_proxy_ratio": _safe_ratio(
            correction_count, role_utterance_counts["tutor"]
        ),
    }


def _temporary_path(directory: Path, final_name: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{final_name}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    return Path(raw_path)


def _write_responses(frame: pd.DataFrame, path: Path, batch_size: int) -> None:
    table = pa.Table.from_pydict(
        {
            "row_id": frame["row_id"].astype("int64").tolist(),
            "response_id": frame["response_id"].astype(str).tolist(),
            "session_id": frame["session_id"].astype(str).tolist(),
            "learning_objective_id": frame["learning_objective_id"].astype(str).tolist(),
            "learning_objective": frame["learning_objective"].astype(str).tolist(),
            "is_correct": frame["is_correct"].astype("int8").tolist(),
        },
        schema=RESPONSES_SCHEMA,
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=max(batch_size, 1),
    )


def _write_session_batch(writer: pq.ParquetWriter, rows: list[dict[str, object]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=SESSION_VIEWS_SCHEMA))


def _write_session_views(
    transcript_by_session: dict[str, Path],
    path: Path,
    batch_size: int,
    progress: Callable[[str], None] | None,
) -> None:
    session_ids = sorted(transcript_by_session)
    progress_interval = max(1, min(1_000, len(session_ids) // 10 or 1))
    writer = pq.ParquetWriter(
        path,
        SESSION_VIEWS_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    rows: list[dict[str, object]] = []
    try:
        for index, session_id in enumerate(session_ids, start=1):
            rows.append(_session_view(transcript_by_session[session_id], session_id))
            if len(rows) >= batch_size:
                _write_session_batch(writer, rows)
                rows.clear()
            if progress is not None and (
                index == len(session_ids) or index % progress_interval == 0
            ):
                progress(f"cache: processed {index}/{len(session_ids)} sessions")
        _write_session_batch(writer, rows)
    finally:
        writer.close()


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _outputs_match(manifest: dict[str, object], output_dir: Path) -> bool:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        return False
    for name in ("responses", "session_views"):
        metadata = outputs.get(name)
        if not isinstance(metadata, dict):
            return False
        relative_path = metadata.get("path")
        expected_sha256 = metadata.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            return False
        path = output_dir / relative_path
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            return False
    return True


def build_cache(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
    batch_size: int = 64,
    progress: Callable[[str], None] | None = None,
) -> CacheBuildResult:
    """Build or reuse deterministic response and session Parquet caches."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    sources = discover_sources(raw_dir)
    responses, transcript_by_session = _load_and_validate_response_tables(sources)
    if progress is not None:
        progress(
            f"cache: validated {len(responses)} responses and "
            f"{len(transcript_by_session)} sessions"
        )
    source_fingerprint = _source_fingerprint(sources)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    responses_path = destination / "responses.parquet"
    session_views_path = destination / "session_views.parquet"
    manifest_path = destination / "cache_manifest.json"

    existing = _load_manifest(manifest_path)
    if (
        not force
        and existing is not None
        and existing.get("cache_version") == CACHE_VERSION
        and existing.get("sources") == source_fingerprint
        and _outputs_match(existing, destination)
    ):
        if progress is not None:
            progress("cache: unchanged inputs; existing cache is valid")
        return CacheBuildResult(
            status="skipped",
            responses_path=responses_path,
            session_views_path=session_views_path,
            manifest_path=manifest_path,
            response_count=len(responses),
            session_count=len(transcript_by_session),
        )

    temporary_responses = _temporary_path(destination, responses_path.name)
    temporary_sessions = _temporary_path(destination, session_views_path.name)
    temporary_manifest = _temporary_path(destination, manifest_path.name)
    temporary_paths = [temporary_responses, temporary_sessions, temporary_manifest]
    try:
        _write_responses(responses, temporary_responses, batch_size)
        _write_session_views(
            transcript_by_session,
            temporary_sessions,
            batch_size,
            progress,
        )

        manifest = {
            "cache_version": CACHE_VERSION,
            "sources": source_fingerprint,
            "outputs": {
                "responses": {
                    "path": responses_path.name,
                    "row_count": len(responses),
                    "sha256": _sha256_file(temporary_responses),
                    "schema": str(RESPONSES_SCHEMA),
                },
                "session_views": {
                    "path": session_views_path.name,
                    "row_count": len(transcript_by_session),
                    "sha256": _sha256_file(temporary_sessions),
                    "schema": str(SESSION_VIEWS_SCHEMA),
                },
            },
        }
        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_responses, responses_path)
        os.replace(temporary_sessions, session_views_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    if progress is not None:
        progress("cache: build complete")
    return CacheBuildResult(
        status="built",
        responses_path=responses_path,
        session_views_path=session_views_path,
        manifest_path=manifest_path,
        response_count=len(responses),
        session_count=len(transcript_by_session),
    )
