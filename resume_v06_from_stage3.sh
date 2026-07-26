#!/usr/bin/env bash
# Resume v06 clean-room pipeline from Stage 3 onward.
# Stage 1 (semantic store), Stage 1b (determinism check), and Stage 2 (sparse CV)
# already completed successfully on 2026-07-23 (see rebuild_v06.log) - skipped here.
# Stage 3 was killed by the OOM killer at 20:33-20:34 UTC on 2026-07-23; a 4G swapfile
# (/swapfile_v06) has since been added for headroom before retrying.
#
# Assumes ./venv already has (none of this is done by this script - see
# build_v06.sh for the confirmed-on-a-fresh-clone details):
#   python3.12 -m venv venv   (the .python-version-pinned interpreter)
#   ./venv/bin/pip install -r requirements.txt -r requirements-embeddings.txt
#   ./venv/bin/pip install -e .   (installs trace_ace itself; every script in
#     this pipeline except build_cache.py/audit_integrity.py assumes it's
#     already importable rather than adding src/ onto sys.path itself)
set -euo pipefail
echo 800 > /proc/self/oom_score_adj 2>/dev/null || true
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd /opt/trace-the-ace
PY=./venv/bin/python

SEM=data/processed/semantic_store_bge_base_v06
ASSET=assets/bge-base-en-v1.5
RUN=experiments/runs/cleanroom_v06
ROBUST=experiments/runs/robust_cleanroom_v06
ART=models/final_ensemble_cleanroom_v06.joblib
ZIP=submissions/builds/trace_ace_cleanroom_v06.zip
ZIP_B=submissions/builds/trace_ace_cleanroom_v06_dbl.zip
STG=submissions/staging/cleanroom_v06
STG_B=submissions/staging/cleanroom_v06_dbl
SPARSE=data/processed/sparse_store

ts(){ date -u +%H:%M:%S; }

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
echo "[$(ts)] V06_PIPELINE_DONE"
