#!/usr/bin/env python3
"""Privacy-safe local verification for an extracted Trace the Ace submission.

The verifier deliberately never serializes response identifiers, transcript or
objective text, or prediction values into its report.  Child-process output is
captured in a private temporary directory and represented only by byte counts
and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Sequence
import zipfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = PROJECT_ROOT / "submissions/builds/trace_ace_cleanroom_v02.zip"
DEFAULT_FIXTURE = PROJECT_ROOT / "data/interim/smoke_fixture"
DEFAULT_WORK_DIR = PROJECT_ROOT / "submissions/runtime_verification"
DEFAULT_BATCH_SIZES = (1, 17, 128)
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_ARCHIVE_BYTES = 60 * 1024**3
MAX_CAPTURE_BYTES = 64 * 1024**2
REPORT_SCHEMA_VERSION = 1
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)

REQUIRED_MEMBERS = {
    "main.py",
    "model/model.joblib",
    "submission_metadata.json",
    "trace_ace/__init__.py",
    "trace_ace/config.py",
    "trace_ace/ensemble.py",
    "trace_ace/features.py",
    "trace_ace/io.py",
    "trace_ace/provenance.py",
    "trace_ace/semantic.py",
    "trace_ace/sparse.py",
    "LICENSE",
    "NOTICE.txt",
    "THIRD_PARTY_LICENSES/BAAI_BGE_MIT.txt",
    "assets/bge-small-en-v1.5/1_Pooling/config.json",
    "assets/bge-small-en-v1.5/2_Normalize/",
    "assets/bge-small-en-v1.5/README.md",
    "assets/bge-small-en-v1.5/config.json",
    "assets/bge-small-en-v1.5/config_sentence_transformers.json",
    "assets/bge-small-en-v1.5/model.safetensors",
    "assets/bge-small-en-v1.5/modules.json",
    "assets/bge-small-en-v1.5/sentence_bert_config.json",
    "assets/bge-small-en-v1.5/special_tokens_map.json",
    "assets/bge-small-en-v1.5/tokenizer.json",
    "assets/bge-small-en-v1.5/tokenizer_config.json",
    "assets/bge-small-en-v1.5/vocab.txt",
}
PROHIBITED_MEMBER_PARTS = {"__pycache__", ".cache", ".git", ".pytest_cache"}
PROHIBITED_SECRET_NAMES = {
    ".env",
    ".netrc",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
FITTING_SYMBOLS = {"fit", "fit_transform", "partial_fit"}
NETWORK_MODULES = {
    "aiohttp",
    "boto3",
    "botocore",
    "ftplib",
    "http",
    "httpx",
    "paramiko",
    "requests",
    "smtplib",
    "socket",
    "urllib",
    "websocket",
    "websockets",
}
PROCESS_MODULES = {"subprocess"}
LOGGING_MODULES = {"logging"}
NETWORK_CALLS = {
    "create_connection",
    "getaddrinfo",
    "gethostbyname",
    "request",
    "urlopen",
}
LOGGING_CALLS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "print",
    "pprint",
    "warn",
    "warning",
    "write",
}
PROCESS_CALLS = {"Popen", "call", "check_call", "check_output", "popen", "run", "system"}
HOST_PATH_TEXT_RE = re.compile(
    r"(?:^|[\s='\"])/(?:code_execution|home|mnt|opt|root|tmp|workspace)(?:/|$)"
    r"|(?:^|[\s='\"])[A-Za-z]:\\"
)
URL_RE = re.compile(r"(?:https?|ftp)://", re.IGNORECASE)
HOST_PATH_BYTE_MARKERS = (
    b"/code_execution/",
    b"/home/",
    b"/mnt/",
    b"/opt/",
    b"/root/",
    b"/tmp/",
    b"/workspace/",
    b"file://",
    b"C:\\Users\\",
)


class VerificationError(RuntimeError):
    """A report-safe, content-free verification failure."""

    def __init__(self, code: str, stage: str) -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage


class JsonArgumentParser(argparse.ArgumentParser):
    """Avoid argparse echoing potentially sensitive argument values."""

    def error(self, message: str) -> None:  # pragma: no cover - exercised by CLI only
        del message
        raise VerificationError("invalid_arguments", "arguments")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise VerificationError("unsafe_zip_member", "zip")
    pure = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise VerificationError("unsafe_zip_member", "zip")
    canonical = pure.as_posix()
    if info.is_dir():
        canonical = canonical.rstrip("/") + "/"
    if canonical != name:
        raise VerificationError("noncanonical_zip_member", "zip")
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    allowed_types = {0, stat.S_IFREG, stat.S_IFDIR}
    if file_type not in allowed_types or stat.S_ISLNK(unix_mode):
        raise VerificationError("unsupported_zip_member_type", "zip")
    if info.flag_bits & 0x1:
        raise VerificationError("encrypted_zip_member", "zip")
    return canonical


def inspect_zip(zip_path: str | Path) -> tuple[dict[str, object], list[zipfile.ZipInfo]]:
    archive_path = Path(zip_path).resolve()
    if not archive_path.is_file():
        raise VerificationError("zip_not_found", "zip")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise VerificationError("zip_too_large", "zip")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = [_safe_member_name(info) for info in infos]
            if len(names) != len(set(names)):
                raise VerificationError("duplicate_zip_member", "zip")
            if len({name.casefold() for name in names}) != len(names):
                raise VerificationError("case_colliding_zip_member", "zip")
            if any(info.date_time != FIXED_ZIP_TIME for info in infos):
                raise VerificationError("zip_timestamp_not_fixed", "zip")
            for info in infos:
                mode = (info.external_attr >> 16) & 0o777
                expected_mode = 0o755 if info.is_dir() else 0o644
                if mode != expected_mode:
                    raise VerificationError("zip_mode_not_fixed", "zip")
                expected_compression = (
                    zipfile.ZIP_STORED if info.is_dir() else zipfile.ZIP_DEFLATED
                )
                if info.compress_type != expected_compression:
                    raise VerificationError("zip_compression_not_fixed", "zip")
            name_set = set(names)
            if not REQUIRED_MEMBERS.issubset(name_set):
                raise VerificationError("required_zip_member_missing", "zip")
            for name in names:
                parts = PurePosixPath(name.rstrip("/")).parts
                lowered = {part.casefold() for part in parts}
                if parts and parts[0].casefold() == "data":
                    raise VerificationError("packaged_data_detected", "zip")
                if name.casefold() == "submission.csv":
                    raise VerificationError("packaged_prediction_detected", "zip")
                if lowered.intersection(part.casefold() for part in PROHIBITED_MEMBER_PARTS):
                    raise VerificationError("packaged_cache_detected", "zip")
                if parts and parts[-1].casefold() in PROHIBITED_SECRET_NAMES:
                    raise VerificationError("packaged_secret_detected", "zip")
                if parts and (
                    parts[-1].casefold().endswith(".pem")
                    or parts[-1].casefold().endswith(".key")
                ):
                    raise VerificationError("packaged_secret_detected", "zip")
            total_uncompressed = sum(info.file_size for info in infos)
            if total_uncompressed > MAX_ARCHIVE_BYTES:
                raise VerificationError("zip_uncompressed_size_too_large", "zip")
            if archive.testzip() is not None:
                raise VerificationError("zip_crc_failed", "zip")
            member_manifest = []
            for info, name in zip(infos, names, strict=True):
                digest = hashlib.sha256()
                with archive.open(info, "r") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                member_manifest.append(
                    {
                        "bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "mode": f"{((info.external_attr >> 16) & 0o777):04o}",
                        "name": name,
                        "sha256": digest.hexdigest(),
                    }
                )
    except VerificationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as error:
        raise VerificationError("invalid_zip", "zip") from error
    return (
        {
            "bytes": archive_path.stat().st_size,
            "crc_passed": True,
            "entry_count": len(infos),
            "fixed_modes": True,
            "fixed_timestamps": True,
            "members": member_manifest,
            "safe_paths": True,
            "sha256": sha256_file(archive_path),
            "uncompressed_bytes": total_uncompressed,
        },
        infos,
    )


def safe_extract_zip(zip_path: str | Path, destination: str | Path) -> None:
    extraction_root = Path(destination).resolve()
    if extraction_root.exists() and any(extraction_root.iterdir()):
        raise VerificationError("extraction_directory_not_empty", "extraction")
    extraction_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(Path(zip_path).resolve()) as archive:
            for info in archive.infolist():
                name = _safe_member_name(info)
                target = extraction_root.joinpath(*PurePosixPath(name.rstrip("/")).parts)
                if not _is_within(target, extraction_root):
                    raise VerificationError("unsafe_extraction_target", "extraction")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(0o755)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                target.chmod(0o644)
    except VerificationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as error:
        raise VerificationError("safe_extraction_failed", "extraction") from error


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def audit_python_sources(extraction_root: str | Path) -> dict[str, object]:
    root = Path(extraction_root).resolve()
    violations: list[dict[str, object]] = []
    paths = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise VerificationError("python_source_missing", "static_audit")

    def add(path: Path, line: int, category: str) -> None:
        violations.append(
            {
                "category": category,
                "line": int(line),
                "member": path.relative_to(root).as_posix(),
            }
        )

    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=path.relative_to(root).as_posix())
        except (OSError, UnicodeDecodeError, SyntaxError):
            add(path, 0, "unparseable_source")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in FITTING_SYMBOLS:
                    add(path, node.lineno, "fitting")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in NETWORK_MODULES:
                        add(path, node.lineno, "network")
                    elif top in PROCESS_MODULES:
                        add(path, node.lineno, "process_execution")
                    elif top in LOGGING_MODULES:
                        add(path, node.lineno, "logging")
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".", 1)[0]
                if top in NETWORK_MODULES:
                    add(path, node.lineno, "network")
                elif top in PROCESS_MODULES:
                    add(path, node.lineno, "process_execution")
                elif top in LOGGING_MODULES:
                    add(path, node.lineno, "logging")
            elif isinstance(node, ast.Call):
                dotted = _dotted_name(node.func) or ""
                pieces = dotted.split(".")
                root_name = pieces[0] if pieces else ""
                leaf = pieces[-1] if pieces else ""
                if leaf in FITTING_SYMBOLS:
                    add(path, node.lineno, "fitting")
                if leaf in NETWORK_CALLS or root_name in NETWORK_MODULES:
                    add(path, node.lineno, "network")
                if leaf in PROCESS_CALLS and root_name in PROCESS_MODULES.union({"os"}):
                    add(path, node.lineno, "process_execution")
                if leaf in LOGGING_CALLS and (
                    leaf in {"print", "pprint"}
                    or root_name in LOGGING_MODULES
                    or root_name in {"logger", "log", "warnings"}
                    or dotted in {"sys.stderr.write", "sys.stdout.write", "stderr.write", "stdout.write"}
                ):
                    add(path, node.lineno, "logging")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if HOST_PATH_TEXT_RE.search(node.value):
                    add(path, getattr(node, "lineno", 0), "absolute_path")
                if URL_RE.search(node.value):
                    add(path, getattr(node, "lineno", 0), "network")
    unique = {
        (item["member"], item["line"], item["category"]): item for item in violations
    }
    ordered = [unique[key] for key in sorted(unique)]
    return {
        "passed": not ordered,
        "python_file_count": len(paths),
        "violation_count": len(ordered),
        "violations": ordered,
    }


def audit_package_bytes(extraction_root: str | Path) -> dict[str, object]:
    root = Path(extraction_root).resolve()
    violations: list[dict[str, str]] = []
    file_count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        file_count += 1
        carry = b""
        found = False
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                lowered = (carry + chunk).lower()
                if any(marker.lower() in lowered for marker in HOST_PATH_BYTE_MARKERS):
                    found = True
                    break
                carry = lowered[-64:]
        if found:
            violations.append(
                {
                    "category": "absolute_path_bytes",
                    "member": path.relative_to(root).as_posix(),
                }
            )
    return {
        "file_count": file_count,
        "passed": not violations,
        "violation_count": len(violations),
        "violations": violations,
    }


def _asset_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if ".cache" not in path.relative_to(root).parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        kind = "directory" if path.is_dir() else "file"
        for value in (kind, relative):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def audit_submission_metadata(extraction_root: str | Path) -> dict[str, object]:
    """Cross-check safe package metadata against the extracted package bytes."""

    root = Path(extraction_root).resolve()
    metadata_path = root / "submission_metadata.json"
    model_path = root / "model/model.joblib"
    asset_root = root / "assets/bge-small-en-v1.5"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError("submission_metadata_invalid", "provenance") from error
    if not isinstance(metadata, dict):
        raise VerificationError("submission_metadata_invalid", "provenance")
    required_strings = (
        "artifact_sha256",
        "bge_revision",
        "bge_model_sha256",
        "bge_asset_tree_sha256",
        "runtime_commit",
    )
    if any(
        not isinstance(metadata.get(field), str) or not metadata[field]
        for field in required_strings
    ) or metadata.get("external_training_data") != []:
        raise VerificationError("submission_metadata_invalid", "provenance")
    actual = {
        "artifact_sha256": sha256_file(model_path),
        "bge_model_sha256": sha256_file(asset_root / "model.safetensors"),
        "bge_asset_tree_sha256": _asset_tree_sha256(asset_root),
    }
    if any(metadata.get(field) != value for field, value in actual.items()):
        raise VerificationError("submission_metadata_hash_mismatch", "provenance")
    return {
        **{field: metadata[field] for field in required_strings},
        "external_training_data": [],
        "hashes_match_extracted_bytes": True,
        "validated": True,
    }


def _read_fixture_contract(fixture_dir: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    fixture = Path(fixture_dir).resolve()
    required = {
        "submission_format.csv",
        "test_features.csv",
        "test_transcripts",
    }
    if not fixture.is_dir() or not all((fixture / name).exists() for name in required):
        raise VerificationError("fixture_missing", "fixture")
    if any(path.is_symlink() for path in fixture.rglob("*")):
        raise VerificationError("fixture_symlink_detected", "fixture")
    try:
        expected = pd.read_csv(
            fixture / "submission_format.csv",
            dtype={"response_id": "string"},
            keep_default_na=False,
        )
        features = pd.read_csv(
            fixture / "test_features.csv",
            dtype={"response_id": "string", "session_id": "string"},
            keep_default_na=False,
        )
    except Exception as error:
        raise VerificationError("fixture_csv_invalid", "fixture") from error
    if list(expected.columns) != ["response_id", "probability"]:
        raise VerificationError("fixture_format_schema_invalid", "fixture")
    if list(features.columns) != [
        "response_id",
        "session_id",
        "learning_objective_id",
        "learning_objective",
    ]:
        raise VerificationError("fixture_feature_schema_invalid", "fixture")
    expected_ids = expected["response_id"].astype(str)
    feature_ids = features["response_id"].astype(str)
    if (
        expected.empty
        or expected_ids.eq("").any()
        or expected_ids.duplicated().any()
        or feature_ids.eq("").any()
        or feature_ids.duplicated().any()
        or set(expected_ids) != set(feature_ids)
    ):
        raise VerificationError("fixture_identity_invalid", "fixture")
    transcripts = fixture / "test_transcripts"
    if not transcripts.is_dir():
        raise VerificationError("fixture_transcripts_missing", "fixture")
    for session_id in features["session_id"].astype(str).unique():
        if Path(session_id).name != session_id or not (transcripts / f"{session_id}.csv").is_file():
            raise VerificationError("fixture_transcript_invalid", "fixture")
    return expected, {"row_count": len(expected), "validated": True}


def _copy_fixture_read_only(fixture_dir: Path, extraction_root: Path) -> None:
    target = extraction_root / "data"
    if target.exists():
        raise VerificationError("extracted_data_collision", "fixture")
    try:
        shutil.copytree(fixture_dir, target)
        for path in sorted(target.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        target.chmod(0o555)
    except OSError as error:
        raise VerificationError("fixture_copy_failed", "fixture") from error


@contextmanager
def _private_temporary_directory(parent: Path):
    """Yield a mode-700 work tree and reliably clean read-only fixture copies."""

    manager = tempfile.TemporaryDirectory(prefix="trace-ace-verify-", dir=parent)
    root = Path(manager.name).resolve()
    root.chmod(0o700)
    try:
        yield root
    finally:
        # The copied fixture is intentionally read-only during child execution.
        # Restore only paths inside our exact mkdtemp tree before cleanup.
        try:
            for path in root.rglob("*"):
                if path.is_symlink():
                    continue
                path.chmod(0o700 if path.is_dir() else 0o600)
            root.chmod(0o700)
        finally:
            manager.cleanup()


def validate_submission_csv(
    output_path: str | Path,
    expected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(output_path)
    if not path.is_file():
        raise VerificationError("submission_csv_missing", "output_validation")
    try:
        output = pd.read_csv(
            path,
            dtype={"response_id": "string"},
            keep_default_na=False,
        )
    except Exception as error:
        raise VerificationError("submission_csv_invalid", "output_validation") from error
    if list(output.columns) != ["response_id", "probability"]:
        raise VerificationError("submission_schema_invalid", "output_validation")
    ids = output["response_id"].astype(str)
    expected_ids = expected["response_id"].astype(str)
    if len(output) != len(expected):
        raise VerificationError("submission_row_count_invalid", "output_validation")
    if ids.eq("").any() or ids.duplicated().any():
        raise VerificationError("submission_identity_invalid", "output_validation")
    if not ids.reset_index(drop=True).equals(expected_ids.reset_index(drop=True)):
        raise VerificationError("submission_order_invalid", "output_validation")
    try:
        probabilities = pd.to_numeric(output["probability"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise VerificationError("submission_probability_invalid", "output_validation") from error
    if not np.isfinite(probabilities).all():
        raise VerificationError("submission_probability_nonfinite", "output_validation")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise VerificationError("submission_probability_out_of_bounds", "output_validation")
    return output, {
        "bounds_valid": True,
        "columns_valid": True,
        "finite": True,
        "order_valid": True,
        "row_count": len(output),
        "unique_ids": True,
    }


SITE_CUSTOMIZE = r'''"""Verification-only offline and provenance guard."""
import atexit
import json
import os
from pathlib import Path
import socket
import sys

_ROOT = Path(os.environ["TRACE_ACE_VERIFY_EXTRACTION_ROOT"]).resolve()
_PROVENANCE = Path(os.environ["TRACE_ACE_VERIFY_PROVENANCE"]).resolve()

def _blocked(*args, **kwargs):
    del args, kwargs
    raise OSError("network disabled by local submission verifier")

socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.gethostbyname = _blocked

def _audit(event, args):
    del args
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
        raise OSError("network disabled by local submission verifier")

sys.addaudithook(_audit)

@atexit.register
def _write_provenance():
    module_files = []
    invalid = 0
    ensemble_loaded = False
    for name, module in sorted(sys.modules.items()):
        if name != "trace_ace" and not name.startswith("trace_ace."):
            continue
        value = getattr(module, "__file__", None)
        if not value:
            invalid += 1
            continue
        resolved = Path(value).resolve()
        module_files.append(resolved)
        try:
            resolved.relative_to(_ROOT)
        except ValueError:
            invalid += 1
        if name == "trace_ace.ensemble":
            ensemble_loaded = True
    payload = {
        "all_trace_ace_modules_from_extraction": invalid == 0 and bool(module_files),
        "ensemble_loaded_from_extraction": ensemble_loaded and invalid == 0,
        "trace_ace_module_count": len(module_files),
    }
    _PROVENANCE.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
'''


BATCH_RUNNER = r'''import os
from pathlib import Path
from trace_ace.ensemble import predict_test_directory

root = Path.cwd().resolve()
predictions = predict_test_directory(
    data_dir=root / "data",
    artifact_path=root / "model" / "model.joblib",
    bge_asset_path=root / "assets" / "bge-small-en-v1.5",
    runtime_batch_size=int(os.environ["TRACE_ACE_VERIFY_BATCH_SIZE"]),
)
predictions.to_csv(Path(os.environ["TRACE_ACE_VERIFY_OUTPUT"]), index=False)
'''


def _private_stream_evidence(path: Path) -> dict[str, object]:
    warning_hits = 0
    carry = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            lowered = carry + chunk.lower()
            warning_hits += lowered.count(b"warning")
            carry = lowered[-6:]
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "warning_markers": warning_hits,
    }


def _offline_environment(guard_dir: Path, extraction_root: Path, provenance: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "NO_PROXY": "*",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(guard_dir),
            "PYTHONWARNINGS": "default",
            "TOKENIZERS_PARALLELISM": "false",
            "TRACE_ACE_VERIFY_EXTRACTION_ROOT": str(extraction_root),
            "TRACE_ACE_VERIFY_PROVENANCE": str(provenance),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _run_child(
    *,
    command: Sequence[str],
    extraction_root: Path,
    guard_dir: Path,
    run_dir: Path,
    output_path: Path,
    timeout_seconds: float,
    extra_environment: dict[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / "stdout.capture"
    stderr_path = run_dir / "stderr.capture"
    provenance_path = run_dir / "provenance.json"
    output_path.unlink(missing_ok=True)
    environment = _offline_environment(guard_dir, extraction_root, provenance_path)
    if extra_environment:
        environment.update(extra_environment)
    started = time.perf_counter()
    timed_out = False
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=extraction_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = process.wait()
    except OSError as error:
        raise VerificationError("child_launch_failed", "runtime") from error
    runtime_seconds = time.perf_counter() - started
    stdout_evidence = _private_stream_evidence(stdout_path)
    stderr_evidence = _private_stream_evidence(stderr_path)
    if stdout_evidence["bytes"] > MAX_CAPTURE_BYTES or stderr_evidence["bytes"] > MAX_CAPTURE_BYTES:
        raise VerificationError("child_output_limit_exceeded", "runtime")
    if timed_out:
        raise VerificationError("child_timed_out", "runtime")
    if exit_code != 0:
        raise VerificationError("child_exit_nonzero", "runtime")
    if stdout_evidence["bytes"] != 0 or stderr_evidence["bytes"] != 0:
        raise VerificationError("child_emitted_output", "runtime")
    if stderr_evidence["warning_markers"] != 0:
        raise VerificationError("child_emitted_warning", "runtime")
    if not provenance_path.is_file():
        raise VerificationError("runtime_provenance_missing", "provenance")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError("runtime_provenance_invalid", "provenance") from error
    if not provenance.get("all_trace_ace_modules_from_extraction"):
        raise VerificationError("development_tree_import_detected", "provenance")
    if not provenance.get("ensemble_loaded_from_extraction"):
        raise VerificationError("extracted_ensemble_not_loaded", "provenance")
    if not output_path.is_file() or not _is_within(output_path, extraction_root):
        raise VerificationError("runtime_output_missing", "runtime")
    return (
        {
            "exit_code": exit_code,
            "runtime_seconds": round(runtime_seconds, 6),
            "stderr": stderr_evidence,
            "stdout": stdout_evidence,
            "timed_out": False,
        },
        provenance,
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_submission(
    *,
    zip_path: str | Path = DEFAULT_ZIP,
    fixture_dir: str | Path = DEFAULT_FIXTURE,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    python_executable: str | Path = sys.executable,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
) -> dict[str, object]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise VerificationError("timeout_invalid", "arguments")
    normalized_batch_sizes = tuple(dict.fromkeys(int(size) for size in batch_sizes))
    if len(normalized_batch_sizes) < 2 or any(size < 1 for size in normalized_batch_sizes):
        raise VerificationError("batch_sizes_invalid", "arguments")
    # Preserve a virtual-environment launcher symlink.  Resolving it can bypass
    # that environment's site-packages by turning ``venv/bin/python`` into the
    # base interpreter path.
    interpreter = Path(os.path.abspath(os.fspath(python_executable))).expanduser()
    if not interpreter.is_file():
        raise VerificationError("python_executable_missing", "arguments")

    archive_path = Path(zip_path).resolve()
    fixture_path = Path(fixture_dir).resolve()
    working_root = Path(work_dir).resolve()
    if _is_within(working_root, fixture_path):
        raise VerificationError("work_directory_overlaps_fixture", "arguments")
    zip_evidence, _ = inspect_zip(archive_path)
    expected, fixture_evidence = _read_fixture_contract(fixture_path)
    working_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "batch_invariance": {},
        "fixture": fixture_evidence,
        "offline_execution": {
            "network_audit_guard": True,
            "offline_environment": True,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "running",
        "zip": zip_evidence,
    }

    try:
        with _private_temporary_directory(working_root) as temporary_root:
            extraction_root = temporary_root / "extracted"
            guard_dir = temporary_root / "guard"
            guard_dir.mkdir()
            (guard_dir / "sitecustomize.py").write_text(SITE_CUSTOMIZE, encoding="utf-8")
            safe_extract_zip(archive_path, extraction_root)
            source_audit = audit_python_sources(extraction_root)
            byte_audit = audit_package_bytes(extraction_root)
            report["static_audit"] = {
                "package_bytes": byte_audit,
                "python_sources": source_audit,
            }
            if not source_audit["passed"] or not byte_audit["passed"]:
                raise VerificationError("static_audit_failed", "static_audit")
            _copy_fixture_read_only(fixture_path, extraction_root)
            report["extraction"] = {
                "ephemeral": True,
                "safe": True,
            }
            report["submission_metadata"] = audit_submission_metadata(
                extraction_root
            )
            expected_model = extraction_root / "model/model.joblib"
            expected_assets = extraction_root / "assets/bge-small-en-v1.5"
            if not expected_model.is_file() or not expected_assets.is_dir():
                raise VerificationError("packaged_runtime_asset_missing", "provenance")
            report["packaged_inputs"] = {
                "asset_model_sha256": sha256_file(expected_assets / "model.safetensors"),
                "assets_from_extraction": _is_within(expected_assets, extraction_root),
                "ensemble_source_sha256": sha256_file(extraction_root / "trace_ace/ensemble.py"),
                "model_from_extraction": _is_within(expected_model, extraction_root),
                "model_sha256": sha256_file(expected_model),
                "source_from_extraction": True,
            }

            main_runs: list[dict[str, object]] = []
            main_frames: list[pd.DataFrame] = []
            main_hashes: list[str] = []
            for index in (1, 2):
                output_path = extraction_root / "submission.csv"
                execution, provenance = _run_child(
                    command=(str(interpreter), "main.py"),
                    extraction_root=extraction_root,
                    guard_dir=guard_dir,
                    run_dir=temporary_root / f"main-run-{index}",
                    output_path=output_path,
                    timeout_seconds=timeout_seconds,
                )
                frame, validation = validate_submission_csv(output_path, expected)
                output_hash = sha256_file(output_path)
                preserved = temporary_root / f"main-output-{index}.csv"
                shutil.copy2(output_path, preserved)
                main_frames.append(frame)
                main_hashes.append(output_hash)
                main_runs.append(
                    {
                        "execution": execution,
                        "output_sha256": output_hash,
                        "provenance": provenance,
                        "validation": validation,
                    }
                )
            if main_hashes[0] != main_hashes[1] or (
                temporary_root / "main-output-1.csv"
            ).read_bytes() != (temporary_root / "main-output-2.csv").read_bytes():
                raise VerificationError("main_output_not_byte_identical", "repeatability")
            report["main_runs"] = main_runs
            report["repeatability"] = {
                "byte_identical": True,
                "output_sha256": main_hashes[0],
            }

            batch_runs: list[dict[str, object]] = []
            batch_frames: list[pd.DataFrame] = []
            for index, batch_size in enumerate(normalized_batch_sizes, start=1):
                output_path = extraction_root / f"batch-output-{index}.csv"
                execution, provenance = _run_child(
                    command=(str(interpreter), "-c", BATCH_RUNNER),
                    extraction_root=extraction_root,
                    guard_dir=guard_dir,
                    run_dir=temporary_root / f"batch-run-{index}",
                    output_path=output_path,
                    timeout_seconds=timeout_seconds,
                    extra_environment={
                        "TRACE_ACE_VERIFY_BATCH_SIZE": str(batch_size),
                        "TRACE_ACE_VERIFY_OUTPUT": str(output_path),
                    },
                )
                frame, validation = validate_submission_csv(output_path, expected)
                batch_frames.append(frame)
                batch_runs.append(
                    {
                        "batch_size": batch_size,
                        "execution": execution,
                        "output_sha256": sha256_file(output_path),
                        "provenance": provenance,
                        "validation": validation,
                    }
                )

            reference = batch_frames[0]["probability"].to_numpy(dtype=np.float64)
            main_values = main_frames[0]["probability"].to_numpy(dtype=np.float64)
            # CPU transformer kernels can change float32 reduction order with
            # batch shape.  The measured clean-room variation is below 7e-8;
            # 1e-7 is a strict numerical-invariance threshold while rejecting
            # any submission-relevant probability change.
            absolute_tolerance = 1e-7
            relative_tolerance = 1e-7
            all_frames = [main_values] + [
                frame["probability"].to_numpy(dtype=np.float64) for frame in batch_frames[1:]
            ]
            absolute_differences = [np.abs(reference - values) for values in all_frames]
            relative_differences = [
                difference
                / np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
                for difference in absolute_differences
            ]
            if any(
                not np.allclose(
                    reference,
                    values,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                    equal_nan=False,
                )
                for values in all_frames
            ):
                raise VerificationError("batch_predictions_differ", "batch_invariance")
            report["batch_invariance"] = {
                "absolute_tolerance": absolute_tolerance,
                "allclose": True,
                "batch_sizes": list(normalized_batch_sizes),
                "max_absolute_difference": float(
                    max(difference.max(initial=0.0) for difference in absolute_differences)
                ),
                "max_relative_difference": float(
                    max(difference.max(initial=0.0) for difference in relative_differences)
                ),
                "order_invariant": True,
                "relative_tolerance": relative_tolerance,
                "runs": batch_runs,
            }
            report["status"] = "passed"
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError("unexpected_verification_failure", "verification") from error
    return report


def _build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report_path: Path | None = None
    try:
        args = _build_parser().parse_args(argv)
        report_path = args.report
        report = verify_submission(
            zip_path=args.zip_path,
            fixture_dir=args.fixture,
            work_dir=args.work_dir,
            python_executable=args.python_executable,
            timeout_seconds=args.timeout,
            batch_sizes=args.batch_sizes,
        )
        exit_code = 0
    except VerificationError as error:
        report = {
            "failure": {"code": error.code, "stage": error.stage},
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "failed",
        }
        exit_code = 1
    except Exception:
        report = {
            "failure": {"code": "unexpected_cli_failure", "stage": "cli"},
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "failed",
        }
        exit_code = 1
    if report_path is not None:
        try:
            _atomic_write_json(report_path, report)
        except Exception:
            report = {
                "failure": {"code": "report_write_failed", "stage": "report"},
                "schema_version": REPORT_SCHEMA_VERSION,
                "status": "failed",
            }
            exit_code = 1
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
