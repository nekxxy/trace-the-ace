"""Focused fail-closed tests for validation provenance bindings."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_ensemble import _validate_component_evidence
from trace_ace import provenance as provenance_module
from trace_ace.config import BGE_DIMENSION
from trace_ace.features import DENSE_FEATURE_NAMES
from trace_ace.provenance import (
    SEMANTIC_CV_SOURCE_FILES,
    SEMANTIC_TRAINING_SOURCE_FILES,
    SPARSE_CV_SOURCE_FILES,
    SPARSE_TRAINING_SOURCE_FILES,
    VALIDATION_PIPELINE_SOURCE_FILES,
    file_sha256,
    trace_ace_source_sha256,
    validate_store_manifest_source,
)


def _component_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = tmp_path / "manifest.json"
    oof = tmp_path / "oof.parquet"
    runner = tmp_path / "runner.py"
    manifest.write_text('{"store": "synthetic"}\n', encoding="utf-8")
    oof.write_bytes(b"synthetic-oof")
    runner.write_text("# synthetic runner\n", encoding="utf-8")
    config = {"seed": 7}
    source_files = ("config.py",)
    evidence: dict[str, object] = {
        "config": config,
        "response_identity_sha256": "response-identity",
        "response_view_sha256": "response-view",
        "store_manifest_sha256": file_sha256(manifest),
        "training_source_sha256": trace_ace_source_sha256(source_files),
        "runner_source_sha256": file_sha256(runner),
        "oof_sha256": file_sha256(oof),
    }
    arguments: dict[str, object] = {
        "evidence": evidence,
        "name": "synthetic",
        "expected_config": config,
        "response_digest": "response-identity",
        "response_view_hash": "response-view",
        "store_manifest_path": manifest,
        "oof_path": oof,
        "source_files": source_files,
        "runner_path": runner,
    }
    return evidence, arguments


def test_component_evidence_accepts_exact_artifact_hashes(tmp_path: Path) -> None:
    _, arguments = _component_fixture(tmp_path)

    _validate_component_evidence(**arguments)  # type: ignore[arg-type]


def test_component_evidence_rejects_oof_changed_after_metrics(tmp_path: Path) -> None:
    _, arguments = _component_fixture(tmp_path)
    arguments["oof_path"].write_bytes(b"changed-oof")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="oof_sha256 differs from current inputs"):
        _validate_component_evidence(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "response_identity_sha256",
        "response_view_sha256",
        "store_manifest_sha256",
        "training_source_sha256",
        "runner_source_sha256",
    ],
)
def test_component_evidence_rejects_stale_provenance_field(
    tmp_path: Path, field: str
) -> None:
    evidence, arguments = _component_fixture(tmp_path)
    evidence[field] = "stale"

    with pytest.raises(ValueError, match=rf"{field} differs from current inputs"):
        _validate_component_evidence(**arguments)  # type: ignore[arg-type]


def test_source_allowlists_cover_model_and_split_dependencies() -> None:
    assert {"config.py", "sparse.py", "sparse_store.py", "training.py"}.issubset(
        SPARSE_TRAINING_SOURCE_FILES
    )
    assert {
        "config.py",
        "semantic.py",
        "semantic_store.py",
        "semantic_training.py",
    }.issubset(SEMANTIC_TRAINING_SOURCE_FILES)
    assert set(SPARSE_TRAINING_SOURCE_FILES).issubset(SPARSE_CV_SOURCE_FILES)
    assert set(SEMANTIC_TRAINING_SOURCE_FILES).issubset(SEMANTIC_CV_SOURCE_FILES)
    assert "validation.py" in SPARSE_CV_SOURCE_FILES
    assert "validation.py" in SEMANTIC_CV_SOURCE_FILES
    assert {"ensemble.py", *SPARSE_CV_SOURCE_FILES, *SEMANTIC_CV_SOURCE_FILES}.issubset(
        VALIDATION_PIPELINE_SOURCE_FILES
    )


def _current_store_manifest(store: str) -> dict[str, object]:
    source_root = Path(provenance_module.__file__).resolve().parent
    dense_columns = [f"dense__{name}" for name in DENSE_FEATURE_NAMES]
    if store == "sparse":
        code = {
            "builder_sha256": file_sha256(source_root / "sparse_store.py"),
            "transform_sha256": file_sha256(source_root / "sparse.py"),
        }
        source: dict[str, object] = {"code": code, "dense_columns": dense_columns}
    else:
        code = {
            "builder_sha256": file_sha256(source_root / "semantic_store.py"),
            "config_sha256": file_sha256(source_root / "config.py"),
            "encoder_sha256": file_sha256(source_root / "semantic.py"),
            "feature_store_sha256": file_sha256(source_root / "feature_store.py"),
        }
        source = {
            "code": code,
            "dense_columns": dense_columns,
            "embedding_dimension": BGE_DIMENSION,
        }
    return {"source": source}


@pytest.mark.parametrize("store", ["sparse", "semantic"])
def test_store_manifest_source_accepts_current_fingerprints(store: str) -> None:
    validate_store_manifest_source(_current_store_manifest(store), store=store)


@pytest.mark.parametrize("store", ["sparse", "semantic"])
def test_store_manifest_source_rejects_stale_builder(store: str) -> None:
    manifest = _current_store_manifest(store)
    manifest["source"]["code"]["builder_sha256"] = "stale"  # type: ignore[index]

    with pytest.raises(ValueError, match="different feature source code"):
        validate_store_manifest_source(manifest, store=store)
