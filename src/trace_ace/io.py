"""Strict competition-table readers shared by cache and runtime inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TRANSCRIPT_COLUMNS = ["session_id", "utterance_id", "role", "content", "timestamp"]
ALLOWED_ROLES = {"tutor", "student", "background"}


def read_transcript(path: str | Path, *, expected_session_id: str | None = None) -> pd.DataFrame:
    """Read, validate, and stably order one transcript without NA coercion."""

    transcript_path = Path(path)
    frame = pd.read_csv(
        transcript_path,
        dtype={column: "string" for column in TRANSCRIPT_COLUMNS},
        keep_default_na=False,
    )
    if list(frame.columns) != TRANSCRIPT_COLUMNS:
        raise ValueError("invalid transcript schema")
    if frame.empty or frame[TRANSCRIPT_COLUMNS].isna().any().any():
        raise ValueError("invalid transcript rows")
    session_values = frame["session_id"].astype(str)
    if session_values.nunique() != 1:
        raise ValueError("a transcript must contain exactly one session")
    if expected_session_id is not None and session_values.iloc[0] != str(expected_session_id):
        raise ValueError("transcript session does not match its expected identity")
    roles = frame["role"].astype(str).str.casefold()
    if not roles.isin(ALLOWED_ROLES).all():
        raise ValueError("transcript contains an unsupported role")
    frame = frame.copy()
    frame["role"] = roles
    if frame["utterance_id"].duplicated().any():
        raise ValueError("transcript utterance IDs must be unique")
    numeric = pd.to_numeric(frame["utterance_id"], errors="coerce")
    frame["_order"] = numeric if numeric.notna().all() else frame["utterance_id"].astype(str)
    return frame.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
