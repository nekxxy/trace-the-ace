"""Static and synthetic checks for the submission packaging boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import struct
import subprocess
import zipfile

import pytest

from scripts.build_submission import (
    FIXED_ZIP_TIME,
    RUNTIME_MODULES,
    _verify_runtime_checkout,
    _write_zip,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_ROOT = PROJECT_ROOT / "src" / "trace_ace"
BGE_ASSET_ROOT = PROJECT_ROOT / "assets" / "bge-small-en-v1.5"
TRAINING_ONLY_MODULES = {
    "trace_ace.cache",
    "trace_ace.feature_store",
    "trace_ace.semantic_store",
    "trace_ace.semantic_training",
    "trace_ace.sparse_store",
    "trace_ace.training",
    "trace_ace.validation",
}
FITTING_SYMBOLS = {"fit", "fit_transform", "partial_fit"}
HOST_ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s='\"])/(?:code_execution|home|mnt|opt|root|tmp|workspace)(?:/|$)"
    r"|(?:^|[\s='\"])[A-Za-z]:\\"
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_checkout_fixture(root: Path) -> tuple[Path, str]:
    checkout = root / "runtime"
    checkout.mkdir()
    _git(checkout, "init", "--quiet")
    (checkout / "runtime.txt").write_text("pinned\n", encoding="utf-8")
    _git(checkout, "add", "runtime.txt")
    _git(
        checkout,
        "-c",
        "user.name=Trace Ace Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "pinned runtime",
    )
    return checkout, _git(checkout, "rev-parse", "HEAD")


def test_runtime_checkout_guard_accepts_only_exact_clean_tracked_state(
    tmp_path: Path,
) -> None:
    checkout, commit = _runtime_checkout_fixture(tmp_path)
    assert _verify_runtime_checkout(checkout, expected_commit=commit) == commit

    (checkout / "untracked.txt").write_text("allowed\n", encoding="utf-8")
    assert _verify_runtime_checkout(checkout, expected_commit=commit) == commit

    (checkout / "runtime.txt").write_text("modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked changes"):
        _verify_runtime_checkout(checkout, expected_commit=commit)

    _git(checkout, "restore", "runtime.txt")
    (checkout / "runtime.txt").write_text("staged\n", encoding="utf-8")
    _git(checkout, "add", "runtime.txt")
    with pytest.raises(RuntimeError, match="tracked changes"):
        _verify_runtime_checkout(checkout, expected_commit=commit)


def test_runtime_checkout_guard_rejects_wrong_head_and_non_repository(
    tmp_path: Path,
) -> None:
    checkout, commit = _runtime_checkout_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="pinned commit"):
        _verify_runtime_checkout(checkout, expected_commit="0" * len(commit))

    non_repository = tmp_path / "not-a-repository"
    non_repository.mkdir()
    with pytest.raises(RuntimeError, match="not verifiable"):
        _verify_runtime_checkout(non_repository, expected_commit=commit)


def _runtime_source_paths() -> list[Path]:
    return [RUNTIME_SOURCE_ROOT / name for name in RUNTIME_MODULES]


def _json_strings(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)
    elif isinstance(value, str):
        yield value


def test_write_zip_is_byte_deterministic_and_policy_clean(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('synthetic')\n", encoding="utf-8")
    nested = source / "assets" / "model"
    nested.mkdir(parents=True)
    (nested / "weights.bin").write_bytes(bytes(range(64)))
    (source / "empty").mkdir()
    cache = source / ".cache"
    cache.mkdir()
    (cache / "private.txt").write_text("excluded", encoding="utf-8")
    bytecode = source / "__pycache__"
    bytecode.mkdir()
    (bytecode / "main.pyc").write_bytes(b"excluded")

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _write_zip(source, first)
    _write_zip(source, second)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [
            "assets/model/weights.bin",
            "empty/",
            "main.py",
        ]
        for info in archive.infolist():
            assert info.date_time == FIXED_ZIP_TIME
            expected_mode = 0o755 if info.is_dir() else 0o644
            assert (info.external_attr >> 16) & 0o777 == expected_mode


def test_runtime_modules_have_no_fitting_logic_or_training_imports() -> None:
    violations: list[str] = []
    for path in _runtime_source_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in FITTING_SYMBOLS:
                    violations.append(f"{path.name}:{node.lineno}: defines {node.name}")
            elif isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else function.id
                    if isinstance(function, ast.Name)
                    else None
                )
                if name in FITTING_SYMBOLS:
                    violations.append(f"{path.name}:{node.lineno}: calls {name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in TRAINING_ONLY_MODULES:
                        violations.append(
                            f"{path.name}:{node.lineno}: imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module in TRAINING_ONLY_MODULES:
                    violations.append(
                        f"{path.name}:{node.lineno}: imports {node.module}"
                    )
    assert not violations, "\n".join(violations)


def test_packaged_python_has_no_host_absolute_path_literals() -> None:
    paths = [PROJECT_ROOT / "submission_template" / "main.py", *_runtime_source_paths()]
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and HOST_ABSOLUTE_PATH.search(node.value)
            ):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, f"host absolute path literals: {violations}"


def test_bge_metadata_has_no_host_absolute_paths() -> None:
    violations: list[str] = []
    for path in sorted(BGE_ASSET_ROOT.rglob("*.json")):
        if ".cache" in path.relative_to(BGE_ASSET_ROOT).parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(HOST_ABSOLUTE_PATH.search(value) for value in _json_strings(payload)):
            violations.append(path.relative_to(BGE_ASSET_ROOT).as_posix())

    safetensors_path = BGE_ASSET_ROOT / "model.safetensors"
    with safetensors_path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length).decode("utf-8"))
    metadata = header.get("__metadata__", {})
    if any(HOST_ABSOLUTE_PATH.search(value) for value in _json_strings(metadata)):
        violations.append("model.safetensors::__metadata__")

    assert not violations, f"BGE metadata contains host paths: {violations}"
