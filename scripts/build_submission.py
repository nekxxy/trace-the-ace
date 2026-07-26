#!/usr/bin/env python3
"""Build and validate a deterministic code-submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import joblib

from trace_ace.ensemble import FinalEnsembleArtifact
from trace_ace.provenance import asset_tree_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CHECKOUT = PROJECT_ROOT / "vendor/tutoring-outcomes-runtime"
RUNTIME_COMMIT = "ea9a81755e101b8036e386430c3a2f3d7c655f2e"
RUNTIME_MODULES = (
    "__init__.py",
    "calibration.py",
    "config.py",
    "ensemble.py",
    "features.py",
    "io.py",
    "provenance.py",
    "semantic.py",
    "sparse.py",
)
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def _verify_runtime_checkout(
    checkout: str | Path = RUNTIME_CHECKOUT,
    *,
    expected_commit: str = RUNTIME_COMMIT,
) -> str:
    """Require the configured official runtime checkout at a clean pinned HEAD."""

    path = Path(checkout)
    if not path.is_dir():
        raise RuntimeError("official runtime checkout is missing")

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ("git", *arguments),
                cwd=path,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("official runtime checkout is not verifiable") from error
        return result.stdout.strip()

    if git("rev-parse", "--is-inside-work-tree") != "true":
        raise RuntimeError("official runtime checkout is not a Git worktree")
    head = git("rev-parse", "--verify", "HEAD^{commit}")
    if head != expected_commit:
        raise RuntimeError("official runtime checkout is not at the pinned commit")
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise RuntimeError("official runtime checkout has tracked changes")
    return head


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_zip(source: Path, output: Path) -> None:
    entries = sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix())
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in entries:
            relative = path.relative_to(source).as_posix()
            if any(part in {"__pycache__", ".cache"} for part in path.relative_to(source).parts):
                continue
            if path.is_dir():
                if any(path.iterdir()):
                    continue
                info = zipfile.ZipInfo(relative.rstrip("/") + "/", FIXED_ZIP_TIME)
                info.external_attr = (0o755 << 16) | 0x10
                archive.writestr(info, b"")
                continue
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "models/final_ensemble_cleanroom_v02.joblib",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=PROJECT_ROOT / "submissions/staging/cleanroom_v02",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "submissions/builds/trace_ace_cleanroom_v02.zip",
    )
    args = parser.parse_args()
    runtime_commit = _verify_runtime_checkout()
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    artifact = joblib.load(args.artifact)
    if not isinstance(artifact, FinalEnsembleArtifact):
        raise RuntimeError("artifact has the wrong type")
    artifact.validate_runtime()
    resources = json.loads((PROJECT_ROOT / "configs/resources.json").read_text())
    approved_bge = resources["bge_base_en_v1_5"]
    if args.staging.exists():
        shutil.rmtree(args.staging)
    args.staging.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "submission_template/main.py", args.staging / "main.py")
    shutil.copy2(PROJECT_ROOT / "submission_template/LICENSE", args.staging / "LICENSE")
    shutil.copy2(PROJECT_ROOT / "submission_template/NOTICE.txt", args.staging / "NOTICE.txt")
    shutil.copytree(
        PROJECT_ROOT / "submission_template/THIRD_PARTY_LICENSES",
        args.staging / "THIRD_PARTY_LICENSES",
    )

    package_dir = args.staging / "trace_ace"
    package_dir.mkdir()
    for name in RUNTIME_MODULES:
        shutil.copy2(PROJECT_ROOT / "src/trace_ace" / name, package_dir / name)
    model_dir = args.staging / "model"
    model_dir.mkdir()
    shutil.copy2(args.artifact, model_dir / "model.joblib")
    asset_source = PROJECT_ROOT / "assets/bge-base-en-v1.5"
    asset_tree_hash = asset_tree_sha256(asset_source)
    if asset_tree_hash != approved_bge["asset_tree_sha256"]:
        raise RuntimeError("local BGE asset tree is not the approved resource")
    if asset_tree_hash != artifact.training_metadata.get("bge_asset_tree_sha256"):
        raise RuntimeError("artifact and packaged BGE asset trees differ")
    if _sha256(asset_source / "model.safetensors") != approved_bge["model_safetensors_sha256"]:
        raise RuntimeError("local BGE model weights are not the approved resource")
    asset_target = args.staging / "assets/bge-base-en-v1.5"
    shutil.copytree(
        asset_source,
        asset_target,
        ignore=shutil.ignore_patterns(".cache"),
    )
    (asset_target / "2_Normalize").mkdir(exist_ok=True)
    metadata = {
        "artifact_sha256": _sha256(args.artifact),
        "bge_revision": approved_bge["revision"],
        "bge_model_sha256": _sha256(asset_source / "model.safetensors"),
        "bge_asset_tree_sha256": asset_tree_hash,
        "external_training_data": [],
        "runtime_commit": runtime_commit,
    }
    (args.staging / "submission_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".zip.tmp")
    _write_zip(args.staging, temporary)
    temporary.replace(args.output)
    with zipfile.ZipFile(args.output) as archive:
        names = archive.namelist()
        required = {
            "main.py",
            "model/model.joblib",
            "submission_metadata.json",
            "LICENSE",
            "NOTICE.txt",
            "THIRD_PARTY_LICENSES/BAAI_BGE_MIT.txt",
            "assets/bge-base-en-v1.5/model.safetensors",
            "assets/bge-base-en-v1.5/modules.json",
            "assets/bge-base-en-v1.5/tokenizer.json",
            "assets/bge-base-en-v1.5/2_Normalize/",
            *{f"trace_ace/{name}" for name in RUNTIME_MODULES},
        }
        if not required.issubset(names):
            raise RuntimeError("submission ZIP is missing required root assets")
        if len(names) != len(set(names)):
            raise RuntimeError("submission ZIP contains duplicate names")
        if any(
            name.startswith("/") or ".." in Path(name).parts or "\\" in name
            for name in names
        ):
            raise RuntimeError("submission ZIP contains an unsafe path")
        if any(name.startswith("data/") or "__pycache__" in name or ".cache/" in name for name in names):
            raise RuntimeError("submission ZIP contains prohibited cache/data files")
    if args.output.stat().st_size > 60 * 1024**3:
        raise RuntimeError("submission ZIP exceeds the documented 60 GB limit")
    report = {
        "path": str(args.output),
        "sha256": _sha256(args.output),
        "bytes": args.output.stat().st_size,
        "entries": len(names),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
