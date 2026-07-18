"""Synthetic tests for the privacy-safe submission runtime verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from scripts.verify_submission_runtime import (
    VerificationError,
    audit_python_sources,
    safe_extract_zip,
    validate_submission_csv,
    verify_submission,
)
from scripts.build_submission import _write_zip
from trace_ace.provenance import asset_tree_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SYNTHETIC_MAIN = '''from pathlib import Path
from trace_ace.ensemble import predict_test_directory

ROOT = Path(__file__).resolve().parent

def main():
    predictions = predict_test_directory(
        data_dir=ROOT / "data",
        artifact_path=ROOT / "model" / "model.joblib",
        bge_asset_path=ROOT / "assets" / "bge-small-en-v1.5",
    )
    predictions.to_csv(ROOT / "submission.csv", index=False)

if __name__ == "__main__":
    main()
'''


SYNTHETIC_ENSEMBLE = '''from pathlib import Path
import pandas as pd

def predict_test_directory(*, data_dir, artifact_path, bge_asset_path, runtime_batch_size=128):
    if runtime_batch_size < 1:
        raise ValueError("invalid batch size")
    if not Path(artifact_path).is_file() or not Path(bge_asset_path).is_dir():
        raise FileNotFoundError("packaged input missing")
    Path(artifact_path).read_bytes()
    frame = pd.read_csv(Path(data_dir) / "submission_format.csv")[["response_id"]]
    frame["probability"] = [0.2 + (index % 3) * 0.1 for index in range(len(frame))]
    return frame
'''


def _fixture(root: Path) -> tuple[Path, str, str]:
    fixture = root / "fixture"
    transcripts = fixture / "test_transcripts"
    transcripts.mkdir(parents=True)
    private_id = "private-response-alpha"
    private_text = "private synthetic transcript content"
    pd.DataFrame(
        {
            "response_id": [private_id, "private-response-beta"],
            "probability": [0.5, 0.5],
        }
    ).to_csv(fixture / "submission_format.csv", index=False)
    pd.DataFrame(
        {
            "response_id": [private_id, "private-response-beta"],
            "session_id": ["session-one", "session-one"],
            "learning_objective_id": ["objective-one", "objective-two"],
            "learning_objective": ["private objective one", "private objective two"],
        }
    ).to_csv(fixture / "test_features.csv", index=False)
    pd.DataFrame(
        {
            "session_id": ["session-one"],
            "utterance_id": ["1"],
            "role": ["student"],
            "content": [private_text],
            "timestamp": ["00:00:01"],
        }
    ).to_csv(transcripts / "session-one.csv", index=False)
    return fixture, private_id, private_text


def _package(root: Path, *, main_source: str = SYNTHETIC_MAIN) -> Path:
    source = root / "package"
    (source / "trace_ace").mkdir(parents=True)
    (source / "model").mkdir()
    (source / "assets/bge-small-en-v1.5").mkdir(parents=True)
    (source / "THIRD_PARTY_LICENSES").mkdir()
    (source / "main.py").write_text(main_source, encoding="utf-8")
    (source / "trace_ace/__init__.py").write_text("", encoding="utf-8")
    (source / "trace_ace/ensemble.py").write_text(SYNTHETIC_ENSEMBLE, encoding="utf-8")
    for name in ("config.py", "features.py", "io.py", "provenance.py", "semantic.py", "sparse.py"):
        (source / "trace_ace" / name).write_text("", encoding="utf-8")
    (source / "model/model.joblib").write_bytes(b"synthetic-model")
    (source / "assets/bge-small-en-v1.5/model.safetensors").write_bytes(b"weights")
    (source / "assets/bge-small-en-v1.5/modules.json").write_text("{}\n", encoding="utf-8")
    (source / "assets/bge-small-en-v1.5/1_Pooling").mkdir()
    (source / "assets/bge-small-en-v1.5/2_Normalize").mkdir()
    for name in (
        "README.md",
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        (source / "assets/bge-small-en-v1.5" / name).write_text("{}\n", encoding="utf-8")
    (source / "assets/bge-small-en-v1.5/1_Pooling/config.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (source / "LICENSE").write_text("synthetic license\n", encoding="utf-8")
    (source / "NOTICE.txt").write_text("synthetic notice\n", encoding="utf-8")
    (source / "THIRD_PARTY_LICENSES/BAAI_BGE_MIT.txt").write_text(
        "synthetic third-party license\n", encoding="utf-8"
    )
    model_sha256 = hashlib.sha256(b"synthetic-model").hexdigest()
    weights_sha256 = hashlib.sha256(b"weights").hexdigest()
    (source / "submission_metadata.json").write_text(
        json.dumps(
            {
                "artifact_sha256": model_sha256,
                "bge_asset_tree_sha256": asset_tree_sha256(
                    source / "assets/bge-small-en-v1.5"
                ),
                "bge_model_sha256": weights_sha256,
                "bge_revision": "synthetic-revision",
                "external_training_data": [],
                "runtime_commit": "synthetic-runtime",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive = root / "submission.zip"
    _write_zip(source, archive)
    return archive


def test_verifier_runs_extracted_package_and_reports_only_safe_evidence(tmp_path: Path) -> None:
    fixture, private_id, private_text = _fixture(tmp_path)
    archive = _package(tmp_path)

    report = verify_submission(
        zip_path=archive,
        fixture_dir=fixture,
        work_dir=tmp_path / "work",
        python_executable=sys.executable,
        timeout_seconds=30,
        batch_sizes=(1, 2, 7),
    )

    serialized = json.dumps(report, sort_keys=True)
    assert report["status"] == "passed"
    assert report["zip"]["crc_passed"] is True
    assert report["zip"]["fixed_modes"] is True
    assert report["zip"]["fixed_timestamps"] is True
    assert len(report["zip"]["members"]) == report["zip"]["entry_count"]
    assert report["submission_metadata"]["hashes_match_extracted_bytes"] is True
    assert report["repeatability"]["byte_identical"] is True
    assert report["batch_invariance"]["allclose"] is True
    assert all(
        run["provenance"]["all_trace_ace_modules_from_extraction"]
        for run in report["main_runs"]
    )
    assert private_id not in serialized
    assert private_text not in serialized
    assert not any((tmp_path / "work").iterdir())


def test_cli_failure_is_json_and_does_not_echo_captured_content(tmp_path: Path) -> None:
    fixture, private_id, private_text = _fixture(tmp_path)
    emitted_secret = "runtime-private-output-marker"
    archive = _package(
        tmp_path,
        main_source=SYNTHETIC_MAIN.replace(
            "def main():", f"def main():\n    print({emitted_secret!r})"
        ),
    )
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/verify_submission_runtime.py"),
            "--zip",
            str(archive),
            "--fixture",
            str(fixture),
            "--work-dir",
            str(tmp_path / "work"),
            "--batch-sizes",
            "1",
            "2",
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    combined = result.stdout + result.stderr + report_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert result.stderr == ""
    assert payload == written
    assert payload["status"] == "failed"
    assert payload["failure"]["code"] == "static_audit_failed"
    assert private_id not in combined
    assert private_text not in combined
    assert emitted_secret not in combined


def test_static_audit_classifies_prohibited_runtime_behaviors(tmp_path: Path) -> None:
    source = tmp_path / "main.py"
    source.write_text(
        "import requests\n"
        "import subprocess\n"
        "import logging\n"
        "def fit():\n"
        "    print('/opt/private/model')\n"
        "    requests.get('https://example.invalid')\n"
        "    subprocess.run(['true'])\n"
        "    logging.info('x')\n",
        encoding="utf-8",
    )
    result = audit_python_sources(tmp_path)
    categories = {item["category"] for item in result["violations"]}
    assert result["passed"] is False
    assert {
        "absolute_path",
        "fitting",
        "logging",
        "network",
        "process_execution",
    }.issubset(categories)


def test_safe_extraction_rejects_traversal_and_output_validation_is_fail_closed(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape", b"blocked")
    with pytest.raises(VerificationError, match="unsafe_zip_member"):
        safe_extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()

    expected = pd.DataFrame(
        {"response_id": ["first", "second"], "probability": [0.5, 0.5]}
    )
    invalid = tmp_path / "invalid.csv"
    pd.DataFrame(
        {"response_id": ["second", "first"], "probability": [0.2, 1.2]}
    ).to_csv(invalid, index=False)
    with pytest.raises(VerificationError) as captured:
        validate_submission_csv(invalid, expected)
    assert captured.value.code in {"submission_order_invalid", "submission_probability_out_of_bounds"}
