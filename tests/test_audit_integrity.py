from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_integrity", PROJECT_ROOT / "scripts" / "audit_integrity.py"
)
assert SPEC is not None and SPEC.loader is not None
audit_integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_integrity)


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
