#!/usr/bin/env bash
# Full v07 clean-room build under bge-large-en-v1.5 (1024-dim), the next size up
# in the same encoder family that transferred cleanly for v06 (bge-small -> bge-base).
# Serial pipeline: store -> sparse CV -> semantic CV -> primary eval -> robust CV
# -> train_final -> build_submission (double build). Fails closed (set -e).
set -euo pipefail
echo 800 > /proc/self/oom_score_adj 2>/dev/null || true
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd /opt/trace-the-ace
PY=./venv/bin/python

SPARSE=data/processed/sparse_store
SEM=data/processed/semantic_store_bge_large_v07
ASSET=assets/bge-large-en-v1.5
RUN=experiments/runs/cleanroom_v07
ROBUST=experiments/runs/robust_cleanroom_v07
ART=models/final_ensemble_cleanroom_v07.joblib
ZIP=submissions/builds/trace_ace_cleanroom_v07.zip
ZIP_B=submissions/builds/trace_ace_cleanroom_v07_dbl.zip
STG=submissions/staging/cleanroom_v07
STG_B=submissions/staging/cleanroom_v07_dbl

ts(){ date -u +%H:%M:%S; }
echo "[$(ts)] STAGE1: build bge-large semantic store (re-embed, ~12-14h expected)"
$PY scripts/build_semantic_store.py --asset "$ASSET" --output-dir "$SEM" --encode-batch-size 8 --force

echo "[$(ts)] STAGE2: sparse CV (objective-disjoint, unchanged config vs v06)"
$PY scripts/run_sparse_cv.py --store-dir "$SPARSE" --run-dir "$RUN"
echo "[$(ts)] STAGE3: semantic CV (objective-disjoint, bge-large)"
$PY scripts/run_semantic_cv.py --store-dir "$SEM" --run-dir "$RUN"
echo "[$(ts)] STAGE4: primary ensemble eval"
$PY scripts/evaluate_ensemble.py --run-dir "$RUN" --sparse-store "$SPARSE" --semantic-store "$SEM"
echo "[$(ts)] STAGE5: robust CV (all protocols)"
$PY scripts/run_robust_cv.py --sparse-store "$SPARSE" --semantic-store "$SEM" --run-dir "$ROBUST"
echo "[$(ts)] STAGE6: train_final -> v07 artifact"
$PY scripts/train_final.py --sparse-store "$SPARSE" --semantic-store "$SEM" --bge-asset "$ASSET" \
    --output "$ART" --primary-metrics "$RUN/ensemble_metrics.json" --robust-summary "$ROBUST/summary.json"
echo "[$(ts)] STAGE7: build submission ZIP (build A)"
$PY scripts/build_submission.py --artifact "$ART" --staging "$STG" --output "$ZIP"
echo "[$(ts)] STAGE7b: build submission ZIP (build B, byte-identity check)"
$PY scripts/build_submission.py --artifact "$ART" --staging "$STG_B" --output "$ZIP_B"
$PY - <<PYEOF
import sys; sys.path.insert(0,'src')
from trace_ace.provenance import file_sha256
a=file_sha256("$ZIP"); b=file_sha256("$ZIP_B")
print(f"  ZIP A sha256: {a}")
print(f"  ZIP B sha256: {b}")
print("ZIP_DETERMINISTIC" if a==b else "ZIP_NONDETERMINISTIC")
import os
print(f"  ZIP bytes: {os.path.getsize('$ZIP')}")
PYEOF
echo "[$(ts)] V07_PIPELINE_DONE"
