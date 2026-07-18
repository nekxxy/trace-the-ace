#!/usr/bin/env python3
"""Fit all final components and serialize one runtime-aligned artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import scipy
import sklearn

from trace_ace.config import ModelConfig, SEMANTIC_PROTOCOLS
from trace_ace.ensemble import (
    ARTIFACT_FORMAT_VERSION,
    RUNTIME_SOURCE_FILES,
    FinalEnsembleArtifact,
    sha256_file,
)
from trace_ace.provenance import asset_tree_sha256, file_sha256, selected_source_sha256
from trace_ace.semantic_training import train_final_semantic
from trace_ace.training import train_final_sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sparse-store",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sparse_store",
    )
    parser.add_argument(
        "--semantic-store",
        type=Path,
        default=PROJECT_ROOT / "data/processed/semantic_store",
    )
    parser.add_argument(
        "--response-view-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/interim/response_views/response_views_manifest.json",
    )
    parser.add_argument(
        "--bge-asset",
        type=Path,
        default=PROJECT_ROOT / "assets/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models/final_ensemble_cleanroom_v02.joblib",
    )
    parser.add_argument(
        "--primary-metrics",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/cleanroom_v02/ensemble_metrics.json",
    )
    parser.add_argument(
        "--robust-summary",
        type=Path,
        default=PROJECT_ROOT / "experiments/runs/robust_cleanroom_v02/summary.json",
    )
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    config = ModelConfig(sparse_epochs=args.epochs)
    sparse_manifest_path = args.sparse_store / "manifest.json"
    semantic_manifest_path = args.semantic_store / "manifest.json"
    sparse_manifest = json.loads(sparse_manifest_path.read_text(encoding="utf-8"))
    semantic_manifest = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
    response_view_manifest = json.loads(
        args.response_view_manifest.read_text(encoding="utf-8")
    )
    if sparse_manifest["source"]["config"] != config.to_dict():
        raise ValueError("sparse store configuration differs from final training")
    if semantic_manifest["source"]["config"] != config.to_dict():
        raise ValueError("semantic store configuration differs from final training")
    identities = {
        sparse_manifest["source"]["response_identity_sha256"],
        semantic_manifest["source"]["response_identity_sha256"],
    }
    if len(identities) != 1:
        raise ValueError("sparse and semantic stores contain different response rows")
    response_identity = next(iter(identities))
    primary_metrics = json.loads(args.primary_metrics.read_text(encoding="utf-8"))
    robust_summary = json.loads(args.robust_summary.read_text(encoding="utf-8"))
    if primary_metrics.get("config") != config.to_dict():
        raise ValueError("primary validation configuration differs from final training")
    if robust_summary.get("config") != config.to_dict():
        raise ValueError("robust validation configuration differs from final training")
    if primary_metrics.get("response_identity_sha256") != response_identity:
        raise ValueError("primary validation response identity differs from model stores")
    if robust_summary.get("response_identity_sha256") != response_identity:
        raise ValueError("robust validation response identity differs from model stores")
    if _gate_passed(primary_metrics) is not True:
        raise ValueError("primary validation promotion gate did not pass")
    if _gate_passed(robust_summary) is not True:
        raise ValueError("robust validation promotion gate did not pass")
    expected_protocols = {name for name, _, _ in SEMANTIC_PROTOCOLS}
    protocol_metrics = robust_summary.get("protocols")
    if not isinstance(protocol_metrics, dict) or set(protocol_metrics) != expected_protocols:
        raise ValueError("robust validation does not contain every required protocol")
    if not all(_gate_passed(record) for record in protocol_metrics.values()):
        raise ValueError("a required robust protocol promotion gate did not pass")
    for evidence_name, evidence in (
        ("primary", primary_metrics),
        *sorted(protocol_metrics.items()),
    ):
        _validate_metric_evidence(evidence, evidence_name)
    response_view_hash = response_view_manifest["output_sha256"]
    if sparse_manifest["source"]["source_sha256"] != response_view_hash:
        raise ValueError("sparse store is not derived from the current response views")
    if semantic_manifest["source"]["source_sha256"] != response_view_hash:
        raise ValueError("semantic store is not derived from the current response views")
    bge_tree_hash = asset_tree_sha256(args.bge_asset)
    if semantic_manifest["source"]["asset_tree_sha256"] != bge_tree_hash:
        raise ValueError("semantic store and current BGE assets differ")
    feature_config = response_view_manifest["sources"]["feature_config"]
    sparse_model = train_final_sparse(
        store_dir=args.sparse_store,
        config=config,
        progress=lambda message: print(message, flush=True),
    )
    semantic_model = train_final_semantic(
        args.semantic_store,
        config=config,
        progress=lambda message: print(message, flush=True),
    )
    artifact = FinalEnsembleArtifact(
        format_version=ARTIFACT_FORMAT_VERSION,
        config=config.to_dict(),
        feature_config=feature_config,
        sparse=sparse_model,
        semantic=semantic_model,
        weights=(config.full_weight, config.role_weight, config.semantic_weight),
        training_metadata={
            "python": platform.python_version(),
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "external_training_data": [],
            "response_identity_sha256": response_identity,
            "response_views_sha256": response_view_hash,
            "sparse_store_manifest_sha256": file_sha256(sparse_manifest_path),
            "semantic_store_manifest_sha256": file_sha256(semantic_manifest_path),
            "primary_validation_sha256": file_sha256(args.primary_metrics),
            "robust_validation_sha256": file_sha256(args.robust_summary),
            "bge_asset_tree_sha256": bge_tree_hash,
            "runtime_source_sha256": selected_source_sha256(
                PROJECT_ROOT / "src/trace_ace", RUNTIME_SOURCE_FILES
            ),
            "training_source_sha256": selected_source_sha256(
                PROJECT_ROOT / "src/trace_ace",
                (
                    "training.py",
                    "semantic_training.py",
                    "sparse_store.py",
                    "semantic_store.py",
                    "validation.py",
                ),
            ),
        },
    )
    artifact.validate_runtime()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".joblib.tmp")
    joblib.dump(artifact, temporary, compress=3)
    restored = joblib.load(temporary)
    if not isinstance(restored, FinalEnsembleArtifact):
        raise RuntimeError("serialized final artifact has the wrong type")
    restored.validate_runtime()
    temporary.replace(args.output)
    metadata = {
        "artifact": args.output.name,
        "sha256": sha256_file(args.output),
        "bytes": args.output.stat().st_size,
        "config": config.to_dict(),
        "training_metadata": artifact.training_metadata,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact": str(args.output), "sha256": metadata["sha256"]}))
    return 0


def _gate_passed(evidence: object) -> bool:
    return bool(
        isinstance(evidence, dict)
        and isinstance(evidence.get("promotion_gate"), dict)
        and evidence["promotion_gate"].get("passed") is True
    )


def _validate_metric_evidence(evidence: object, name: str) -> None:
    if not isinstance(evidence, dict):
        raise ValueError(f"{name} validation evidence is not an object")
    for component in ("baseline", "full", "role", "semantic", "ensemble"):
        record = evidence.get(component)
        if not isinstance(record, dict):
            raise ValueError(f"{name} validation lacks {component} metrics")
        values = np.asarray(
            [record.get(metric) for metric in ("log_loss", "roc_auc", "brier", "ece10")],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} validation contains non-finite metrics")
    folds = evidence.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError(f"{name} validation must contain five fold records")


if __name__ == "__main__":
    raise SystemExit(main())
