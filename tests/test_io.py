from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trace_ace.io import TRANSCRIPT_COLUMNS, read_transcript


def _transcript_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["session-a", "10", "STUDENT", "null", "00:00:10"],
            ["session-a", "2", "Tutor", "NA", "00:00:02"],
            ["session-a", "1", "Background", "NaN", "00:00:01"],
        ],
        columns=TRANSCRIPT_COLUMNS,
    )


def _write_transcript(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "session-a.csv"
    frame.to_csv(path, index=False)
    return path


def test_reader_preserves_na_like_content_and_sorts_numeric_ids(
    tmp_path: Path,
) -> None:
    path = _write_transcript(tmp_path, _transcript_frame())

    first = read_transcript(path, expected_session_id="session-a")
    second = read_transcript(path, expected_session_id="session-a")

    pd.testing.assert_frame_equal(first, second)
    assert first["utterance_id"].tolist() == ["1", "2", "10"]
    assert first["content"].tolist() == ["NaN", "NA", "null"]
    assert first["role"].tolist() == ["background", "tutor", "student"]


def test_reader_rejects_wrong_column_order(tmp_path: Path) -> None:
    frame = _transcript_frame()[
        ["utterance_id", "session_id", "role", "content", "timestamp"]
    ]
    path = _write_transcript(tmp_path, frame)

    with pytest.raises(ValueError, match="invalid transcript schema"):
        read_transcript(path)


def test_reader_rejects_expected_session_mismatch(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path, _transcript_frame())

    with pytest.raises(ValueError, match="does not match its expected identity"):
        read_transcript(path, expected_session_id="different-session")


def test_reader_rejects_multiple_sessions(tmp_path: Path) -> None:
    frame = _transcript_frame()
    frame.loc[0, "session_id"] = "session-b"
    path = _write_transcript(tmp_path, frame)

    with pytest.raises(ValueError, match="exactly one session"):
        read_transcript(path)


def test_reader_rejects_duplicate_utterance_ids(tmp_path: Path) -> None:
    frame = _transcript_frame()
    frame.loc[0, "utterance_id"] = "2"
    path = _write_transcript(tmp_path, frame)

    with pytest.raises(ValueError, match="utterance IDs must be unique"):
        read_transcript(path)


def test_reader_rejects_unsupported_roles(tmp_path: Path) -> None:
    frame = _transcript_frame()
    frame.loc[0, "role"] = "system"
    path = _write_transcript(tmp_path, frame)

    with pytest.raises(ValueError, match="unsupported role"):
        read_transcript(path)
