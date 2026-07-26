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
from trace_ace.provenance import (
    SEMANTIC_CV_SOURCE_FILES,
    SPARSE_CV_SOURCE_FILES,
    VALIDATION_PIPELINE_SOURCE_FILES,
    asset_tree_sha256,
    file_sha256,
    selected_source_sha256,
    trace_ace_source_sha256,
    validate_store_manifest_source,
)
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
        default=PROJECT_ROOT / "assets/bge-base-en-v1.5",
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
    validate_store_manifest_source(sparse_manifest, store="sparse")
    validate_store_manifest_source(semantic_manifest, store="semantic")
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
    sparse_manifest_hash = file_sha256(sparse_manifest_path)
    semantic_manifest_hash = file_sha256(semantic_manifest_path)
    validation_source_hash = trace_ace_source_sha256(
        VALIDATION_PIPELINE_SOURCE_FILES
    )
    common_provenance = {
        "response_view_sha256": response_view_hash,
        "sparse_store_manifest_sha256": sparse_manifest_hash,
        "semantic_store_manifest_sha256": semantic_manifest_hash,
        "validation_source_sha256": validation_source_hash,
    }
    _require_evidence_fields(primary_metrics, common_provenance, "primary")
    _require_evidence_fields(robust_summary, common_provenance, "robust")
    if primary_metrics.get("weights") != [
        config.full_weight,
        config.role_weight,
        config.semantic_weight,
    ]:
        raise ValueError("primary validation weights differ from final training")
    primary_run_dir = args.primary_metrics.parent
    component_evidence = {
        "sparse": (
            primary_run_dir / "sparse_metrics.json",
            sparse_manifest_hash,
            SPARSE_CV_SOURCE_FILES,
            PROJECT_ROOT / "scripts/run_sparse_cv.py",
            primary_run_dir / "sparse_oof.parquet",
        ),
        "semantic": (
            primary_run_dir / "semantic_metrics.json",
            semantic_manifest_hash,
            SEMANTIC_CV_SOURCE_FILES,
            PROJECT_ROOT / "scripts/run_semantic_cv.py",
            primary_run_dir / "semantic_oof.parquet",
        ),
    }
    for component_name, (
        metrics_path,
        store_manifest_hash,
        source_files,
        runner_path,
        oof_path,
    ) in component_evidence.items():
        evidence = json.loads(metrics_path.read_text(encoding="utf-8"))
        _require_evidence_fields(
            evidence,
            {
                "config": config.to_dict(),
                "response_identity_sha256": response_identity,
                "response_view_sha256": response_view_hash,
                "store_manifest_sha256": store_manifest_hash,
                "training_source_sha256": trace_ace_source_sha256(source_files),
                "runner_source_sha256": file_sha256(runner_path),
            },
            component_name,
        )
        _require_evidence_file_hashes(
            evidence,
            {"oof_sha256": oof_path},
            component_name,
        )
    primary_files = {
        "sparse_metrics_sha256": primary_run_dir / "sparse_metrics.json",
        "semantic_metrics_sha256": primary_run_dir / "semantic_metrics.json",
        "sparse_oof_sha256": primary_run_dir / "sparse_oof.parquet",
        "semantic_oof_sha256": primary_run_dir / "semantic_oof.parquet",
    }
    _require_evidence_file_hashes(primary_metrics, primary_files, "primary")
    if primary_metrics.get("evaluator_source_sha256") != file_sha256(
        PROJECT_ROOT / "scripts/evaluate_ensemble.py"
    ):
        raise ValueError("primary evaluator source differs from current code")
    if robust_summary.get("runner_source_sha256") != file_sha256(
        PROJECT_ROOT / "scripts/run_robust_cv.py"
    ):
        raise ValueError("robust runner source differs from current code")
    for protocol_name, evidence in protocol_metrics.items():
        _require_evidence_fields(
            evidence,
            {
                "protocol": protocol_name,
                "response_identity_sha256": response_identity,
                **common_provenance,
                "runner_source_sha256": robust_summary["runner_source_sha256"],
            },
            protocol_name,
        )
        _require_evidence_file_hashes(
            evidence,
            {"oof_sha256": args.robust_summary.parent / protocol_name / "oof.parquet"},
            protocol_name,
        )
    calibration_evidence = primary_metrics.get("full_calibration")
    if (
        not isinstance(calibration_evidence, list)
        or len(calibration_evidence) != 2
        or not all(
            isinstance(value, (int, float)) and np.isfinite(value)
            for value in calibration_evidence
        )
        or float(calibration_evidence[0]) <= 0.0
    ):
        raise ValueError("primary validation lacks a valid full calibration")
    full_calibration = (
        float(calibration_evidence[0]),
        float(calibration_evidence[1]),
    )
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
        full_calibration=full_calibration,
        training_metadata={
            "python": platform.python_version(),
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "external_training_data": [],
            "response_identity_sha256": response_identity,
            "response_views_sha256": response_view_hash,
            "sparse_store_manifest_sha256": sparse_manifest_hash,
            "semantic_store_manifest_sha256": semantic_manifest_hash,
            "primary_validation_sha256": file_sha256(args.primary_metrics),
            "robust_validation_sha256": file_sha256(args.robust_summary),
            "bge_asset_tree_sha256": bge_tree_hash,
            "runtime_source_sha256": selected_source_sha256(
                PROJECT_ROOT / "src/trace_ace", RUNTIME_SOURCE_FILES
            ),
            "training_source_sha256": validation_source_hash,
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
    components = ("baseline", "full", "role", "semantic", "ensemble")
    log_losses: dict[str, float] = {}
    for component in components:
        record = evidence.get(component)
        if not isinstance(record, dict):
            raise ValueError(f"{name} validation lacks {component} metrics")
        values = np.asarray(
            [record.get(metric) for metric in ("log_loss", "roc_auc", "brier", "ece10")],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} validation contains non-finite metrics")
        log_losses[component] = float(values[0])
    folds = evidence.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError(f"{name} validation must contain five fold records")
    if [record.get("fold") if isinstance(record, dict) else None for record in folds] != list(
        range(5)
    ):
        raise ValueError(f"{name} validation fold identities are invalid")
    for fold in folds:
        for component in components:
            record = fold.get(component)
            if not isinstance(record, dict):
                raise ValueError(f"{name} validation fold lacks {component} metrics")
            values = np.asarray(
                [record.get(metric) for metric in ("log_loss", "roc_auc", "brier", "ece10")],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise ValueError(f"{name} validation fold contains non-finite metrics")
    if not all(
        log_losses["ensemble"] < log_losses[component]
        for component in ("baseline", "full", "role", "semantic")
    ):
        raise ValueError(f"{name} validation metrics contradict the promotion gate")


def _require_evidence_fields(
    evidence: object,
    expected: dict[str, object],
    name: str,
) -> None:
    if not isinstance(evidence, dict):
        raise ValueError(f"{name} validation evidence is not an object")
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ValueError(f"{name} validation {field} differs from current inputs")


def _require_evidence_file_hashes(
    evidence: object,
    files: dict[str, Path],
    name: str,
) -> None:
    if not isinstance(evidence, dict):
        raise ValueError(f"{name} validation evidence is not an object")
    for field, path in files.items():
        if evidence.get(field) != file_sha256(path):
            raise ValueError(f"{name} validation {field} differs from current artifact")


if __name__ == "__main__":
    raise SystemExit(main())
