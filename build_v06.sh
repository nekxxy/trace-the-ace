#!/usr/bin/env bash
# Full v06 clean-room build under bge-base-en-v1.5 (768-dim), reconstructed from
# build_v07.sh's pattern (build_v06.sh itself was never committed - confirmed
# via git log). Serial pipeline: store -> sparse CV -> semantic CV -> primary
# eval -> robust CV -> train_final -> build_submission (double build).
# Fails closed (set -e).
#
# Assumes ./venv already has (none of this is done by this script - confirmed
# on a fresh clone that all three are needed, not just documentation gaps):
#   python3.12 -m venv venv   (the .python-version-pinned interpreter - a venv
#     created with a different python3 minor version installs fine but every
#     script below except build_cache.py fails with "No module named
#     trace_ace" at import time, since only build_cache.py and
#     audit_integrity.py defensively insert src/ onto sys.path themselves)
#   ./venv/bin/pip install -r requirements.txt -r requirements-embeddings.txt
#   ./venv/bin/pip install -e .   (installs trace_ace itself; every other
#     script in this pipeline assumes it's already importable)
set -euo pipefail
echo 800 > /proc/self/oom_score_adj 2>/dev/null || true
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=./venv/bin/python

RAW=data/raw
TRANSCRIPTS=data/raw/train_transcripts
CACHE=data/interim/cache
VIEWS=data/interim/response_views
SPARSE=data/processed/sparse_store
SEM=data/processed/semantic_store_bge_base_v06
ASSET=assets/bge-base-en-v1.5
RUN=experiments/runs/cleanroom_v06
ROBUST=experiments/runs/robust_cleanroom_v06
ART=models/final_ensemble_cleanroom_v06.joblib
ZIP=submissions/builds/trace_ace_cleanroom_v06.zip
ZIP_B=submissions/builds/trace_ace_cleanroom_v06_dbl.zip
STG=submissions/staging/cleanroom_v06
STG_B=submissions/staging/cleanroom_v06_dbl

ts(){ date -u +%H:%M:%S; }
echo "[$(ts)] STAGE0a: build response/session cache from data/raw"
$PY scripts/build_cache.py --raw-dir "$RAW" --output-dir "$CACHE"
echo "[$(ts)] STAGE0b: build per-response text/dense feature views"
$PY scripts/build_response_views.py --cache-dir "$CACHE" --transcripts-dir "$TRANSCRIPTS" --output-dir "$VIEWS"
echo "[$(ts)] STAGE0c: build disk-backed sparse store"
$PY scripts/build_sparse_store.py --source "$VIEWS/response_views.parquet" --output-dir "$SPARSE"
echo "[$(ts)] STAGE1: build bge-base semantic store (re-embed, ~2-4h expected)"
$PY scripts/build_semantic_store.py --source "$VIEWS/response_views.parquet" --asset "$ASSET" --output-dir "$SEM" --encode-batch-size 8 --force

echo "[$(ts)] STAGE2: sparse CV (objective-disjoint)"
$PY scripts/run_sparse_cv.py --store-dir "$SPARSE" --run-dir "$RUN"
echo "[$(ts)] STAGE3: semantic CV (objective-disjoint, bge-base)"
$PY scripts/run_semantic_cv.py --store-dir "$SEM" --run-dir "$RUN"
echo "[$(ts)] STAGE4: primary ensemble eval"
$PY scripts/evaluate_ensemble.py --run-dir "$RUN" --sparse-store "$SPARSE" --semantic-store "$SEM"
echo "[$(ts)] STAGE5: robust CV (all protocols)"
$PY scripts/run_robust_cv.py --sparse-store "$SPARSE" --semantic-store "$SEM" --run-dir "$ROBUST"
echo "[$(ts)] STAGE6: train_final -> v06 artifact"
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
import os
print(f"  ZIP bytes: {os.path.getsize('$ZIP')}")
if a != b:
    print("ZIP_NONDETERMINISTIC")
    sys.exit(1)
print("ZIP_DETERMINISTIC")
PYEOF
echo "[$(ts)] STAGE8: verify submission runtime"
$PY scripts/verify_submission_runtime.py --zip "$ZIP"
echo "[$(ts)] V06_PIPELINE_DONE"
