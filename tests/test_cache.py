from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trace_ace.cache import (  # noqa: E402
    CACHE_VERSION,
    RESPONSES_SCHEMA,
    SESSION_VIEWS_SCHEMA,
    build_cache,
)


def _write_synthetic_data(root: Path) -> Path:
    raw_dir = root / "raw"
    transcripts_dir = raw_dir / "Train transcripts"
    transcripts_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "response_id": "r2",
                "session_id": "session_b",
                "learning_objective_id": "lo2",
                "learning_objective": "Second objective",
            },
            {
                "response_id": "r3",
                "session_id": "session_a",
                "learning_objective_id": "lo1",
                "learning_objective": "First objective",
            },
            {
                "response_id": "r1",
                "session_id": "session_a",
                "learning_objective_id": "lo1",
                "learning_objective": "First objective",
            },
        ]
    ).to_csv(raw_dir / "train_features_fixture.csv", index=False)
    pd.DataFrame(
        [
            {"response_id": "r3", "is_correct": 1},
            {"response_id": "r1", "is_correct": 0},
            {"response_id": "r2", "is_correct": 1},
        ]
    ).to_csv(raw_dir / "train_labels_fixture.csv", index=False)

    pd.DataFrame(
        [
            ["session_a", 1, "background", "Prior context.", "00:00:00"],
            ["session_a", 2, "tutor", "What is two plus two?", "00:00:05"],
            ["session_a", 3, "student", "I don't know", "00:00:07"],
            ["session_a", 4, "tutor", "Not quite; try again.", "00:00:09"],
            ["session_a", 5, "student", "Four", "00:00:12"],
            ["session_a", 6, "tutor", "Correct, great work!", "00:00:15"],
        ],
        columns=["session_id", "utterance_id", "role", "content", "timestamp"],
    ).to_csv(transcripts_dir / "session_a.csv", index=False)
    pd.DataFrame(
        [
            ["session_b", 2, "student", "Yes", "00:00:04"],
            ["session_b", 1, "tutor", "Ready?", "00:00:01"],
        ],
        columns=["session_id", "utterance_id", "role", "content", "timestamp"],
    ).to_csv(transcripts_dir / "session_b.csv", index=False)
    return raw_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_cache_schema_role_preservation_and_counts(tmp_path: Path) -> None:
    raw_dir = _write_synthetic_data(tmp_path)
    output_dir = tmp_path / "cache"

    result = build_cache(raw_dir, output_dir, batch_size=1)

    assert result.status == "built"
    responses = pq.read_table(result.responses_path)
    sessions = pq.read_table(result.session_views_path)
    assert responses.schema == RESPONSES_SCHEMA
    assert sessions.schema == SESSION_VIEWS_SCHEMA

    response_frame = responses.to_pandas()
    assert response_frame["response_id"].tolist() == ["r1", "r2", "r3"]
    assert response_frame["row_id"].tolist() == [0, 1, 2]
    assert response_frame.set_index("response_id")["is_correct"].to_dict() == {
        "r1": 0,
        "r2": 1,
        "r3": 1,
    }

    session_frame = sessions.to_pandas().set_index("session_id")
    row = session_frame.loc["session_a"]
    assert "BACKGROUND: Prior context." in row["full_text"]
    assert "TUTOR: What is two plus two?" in row["full_text"]
    assert "STUDENT: I don't know" in row["full_text"]
    assert row["background_text"] == "Prior context."
    assert row["tutor_text"].splitlines()[-1] == "Correct, great work!"
    assert row["student_text"].splitlines()[-1] == "Four"
    assert row["opening_text"].startswith("BACKGROUND: Prior context.")
    assert row["closing_tutor_text"].endswith("Correct, great work!")
    assert row["closing_student_text"].endswith("Four")

    assert row["utterance_count"] == 6
    assert row["tutor_utterance_count"] == 3
    assert row["student_utterance_count"] == 2
    assert row["background_utterance_count"] == 1
    assert row["duration_seconds"] == pytest.approx(15.0)
    assert row["unclear_student_utterance_count"] == 1
    assert row["question_utterance_count"] == 1
    assert row["role_transition_count"] == 5
    assert row["short_student_utterance_count"] == 2
    assert row["feedback_proxy_count"] == 1
    assert row["correction_proxy_count"] == 1
    assert row["tutor_utterance_ratio"] == pytest.approx(0.5)
    assert row["unclear_student_utterance_ratio"] == pytest.approx(0.5)
    assert row["role_transition_ratio"] == pytest.approx(1.0)


def test_cache_is_deterministic_and_unchanged_inputs_skip(tmp_path: Path) -> None:
    raw_dir = _write_synthetic_data(tmp_path)
    output_dir = tmp_path / "cache"

    first = build_cache(raw_dir, output_dir, batch_size=2)
    first_hashes = {
        "responses": _sha256(first.responses_path),
        "sessions": _sha256(first.session_views_path),
        "manifest": _sha256(first.manifest_path),
    }
    second = build_cache(raw_dir, output_dir, batch_size=2)
    assert second.status == "skipped"
    assert first_hashes == {
        "responses": _sha256(second.responses_path),
        "sessions": _sha256(second.session_views_path),
        "manifest": _sha256(second.manifest_path),
    }

    forced = build_cache(raw_dir, output_dir, force=True, batch_size=2)
    assert forced.status == "built"
    assert first_hashes == {
        "responses": _sha256(forced.responses_path),
        "sessions": _sha256(forced.session_views_path),
        "manifest": _sha256(forced.manifest_path),
    }
    manifest = json.loads(forced.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_version"] == CACHE_VERSION
    assert manifest["sources"]["code"]["builder_sha256"]
    assert manifest["sources"]["code"]["reader_sha256"]
    assert manifest["sources"]["features"]["sha256"]
    assert manifest["sources"]["labels"]["sha256"]
    assert manifest["sources"]["transcripts"]["sha256"]


def test_cache_rejects_invalid_schema(tmp_path: Path) -> None:
    raw_dir = _write_synthetic_data(tmp_path)
    labels_path = raw_dir / "train_labels_fixture.csv"
    labels = pd.read_csv(labels_path).rename(columns={"is_correct": "correct"})
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="Invalid train labels schema"):
        build_cache(raw_dir, tmp_path / "cache")


def test_cache_rejects_nonmatching_response_ids(tmp_path: Path) -> None:
    raw_dir = _write_synthetic_data(tmp_path)
    labels_path = raw_dir / "train_labels_fixture.csv"
    labels = pd.read_csv(labels_path)
    labels.loc[0, "response_id"] = "not_in_features"
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="do not match one-to-one"):
        build_cache(raw_dir, tmp_path / "cache")
