# Trace the Ace — agent orientation

Read this first. For the full narrative of the most recent work session (what
was tried, what broke, what was fixed), see `docs/agent_handoff.md`. For the
version-by-version scoreboard and the lessons that generalize, see
`experiments/improvements/README.md` — that file is the source of truth for
"what's the best candidate right now and why."

## Where things stand (2026-07-27)

- **Best real submission: v06** (bge-base semantic encoder), real public log
  loss **0.6073**, rank 20. Local OOF 0.58742.
- **v07** (bge-base → bge-large encoder) was fully built (zip exists,
  deterministic double-build passed) but **regressed on robust CV vs v06** —
  do not submit it. Kept only as a complete record; see the scoreboard.
- **v08** (knowledge-graph-inspired dense features, `src/trace_ace/graph_features.py`)
  was tested and **not adopted** — CV lift (−0.00016 primary, −0.00010
  session-disjoint) is real-direction but 10-15x below the believable-effect
  floor. Don't re-run this exact ablation expecting a different answer; if
  you revisit graph-style features, they need genuinely new signal (e.g. a
  second independently-pretrained embedder), not more transforms of the same
  cached bge embeddings. See the scoreboard and `docs/agent_handoff.md` for
  detail.
- **v09** (H-B: second embedder as an additive 4th head) tried two
  candidates, **neither adopted, no ensemble architecture was ever built**:
  `all-MiniLM-L6-v2` was **killed by the correlation kill-switch**
  (`corr(bge_base_sem_oof, minilm_sem_oof) = 0.876`, above the 0.85
  threshold). Mean-pooled `distilbert-base-uncased` (no contrastive
  fine-tuning at all) **passed** the kill-switch (`corr = 0.834`) but a
  cheap 4-way blend-weight grid search over already-computed OOF (no new
  architecture needed) found its optimal weight is **exactly 0** — no
  ensemble value despite genuine diversity. Don't re-try either exact
  candidate expecting a different answer. **The generalizable lesson: the
  correlation kill-switch is necessary but not sufficient** — check both
  correlation *and* optimal blend weight (via the cheap OOF grid-search
  method, not a full architecture build) before adopting any H-B candidate.
  See `experiments/improvements/v09_hb_killswitch.md`.
- **v06 reproduced end-to-end from raw competition data on 2026-07-27**
  (fresh clean-room rebuild, not just a code read-through) — CV numbers
  matched the scoreboard within rounding and the final zip passed the full
  `verify_submission_runtime.py` gate. Three real bugs were found and fixed
  in the process, all now in the codebase: bge-base's `config.json` *also*
  leaks a host path (not just bge-large's, contradicting what this file used
  to say — see the encoder-swap section below), `asset_tree_sha256` didn't
  exclude the packaging-only `2_Normalize` directory (this would have
  crashed **real inference**, not just packaging, with `RuntimeError: BGE
  asset tree differs from the trained artifact`), and the batch-invariance
  tolerance in `verify_submission_runtime.py` was calibrated on different
  hardware (widened 1e-7 → 5e-7 with the actual cross-CPU measurement
  documented in the code).
- **`config.py`'s BGE_REPOSITORY/REVISION/DIMENSION currently points at
  bge-base (v06), not bge-large (v07)** — it was left pointed at bge-large
  after v07 was built and had to be reverted back once v08's ablation (which
  uses v06's store) crashed with a dimension mismatch. If you find it
  pointing somewhere unexpected, check which candidate is actually "current
  best" in the scoreboard before assuming it's correct.

## Hard constraints (don't relearn these the hard way)

- **Hardware:** ~3.8GB RAM, 2 cores, no GPU. A 4GB swap file
  (`/swapfile_v06`) is active — needed for semantic CV on anything bge-base
  or larger (peak per-fold memory during `fit_semantic_model` can hit ~1.8GB+
  on a 768-dim store, more on 1024-dim). Heavy jobs run one at a time.
- **Validation is objective-disjoint.** The private test has unseen learning
  objectives. Per-objective-ID priors/features do not generalize — this is a
  hard constraint, not a style preference.
- **Local CV is not real score.** Real score ≈ local OOF + ~0.02 "transfer
  gap" for honest/generalizing changes; changes that sharpen the local fit
  without adding real signal widen that gap and score *worse* live (v05 was
  the cautionary example). See the scoreboard for the full gap-model
  analysis before trusting any new local-only improvement.

## When you change the BGE encoder (BGE_REPOSITORY/REVISION/DIMENSION in `src/trace_ace/config.py`)

This bit us during v07. Editing `config.py` alone is **not enough** — grep
the whole repo for the old asset directory name (e.g. `bge-base-en-v1.5`)
before you consider the swap done. Files that hardcode the active asset name
and must move together:

- `scripts/build_submission.py` (asset source/target paths, `resources.json`
  key, REQUIRED_MEMBERS-equivalent set)
- `scripts/verify_submission_runtime.py` (REQUIRED_MEMBERS, asset_root,
  embedded heredoc runtime script, expected_assets)
- `scripts/audit_integrity.py` (asset path)
- `scripts/build_semantic_store.py`, `scripts/train_final.py` (`--asset`/
  `--bge-asset` CLI defaults)
- `src/predict.py` (`--asset` default)
- `submission_template/main.py` (bge_asset_path), `submission_template/NOTICE.txt`
  (encoder name + revision text)
- `configs/resources.json` (add a new `bge_<size>_en_v1_5` entry — don't
  delete older ones, they're a provenance record)
- `tests/test_packaging.py`, `tests/test_verify_submission_runtime.py`,
  `tests/test_audit_integrity.py` (fixture/reference asset paths)

Run `grep -rln "bge-<old-size>-en-v1.5\|bge_<old_size>_en_v1_5" --include="*.py" --include="*.json" --include="*.txt" .`
(excluding `venv/`) and check every hit.

**This cuts both ways: if you abandon a candidate encoder, revert config.py
(and the same file list) back to whatever the actual current-best is.**
Leaving `config.py` pointed at a rejected candidate (as happened after v07
was abandoned but before v08's ablation caught it via a dimension-mismatch
crash) silently breaks anything that relies on `config.py`'s defaults
matching the current-best store.

**Also check the upstream HF model's `config.json` for a leaked host path**
in `_name_or_path` (bge-large's did: `/root/.cache/torch/...`; **bge-base's
does too** — a fresh download on 2026-07-26 showed
`/root/.cache/torch/sentence_transformers/BAAI_bge-base-en/`, contradicting
an earlier claim in this file that bge-base's was clean — don't trust that
claim, verify it yourself on every fresh download) before
building on top of a fresh asset download —
`tests/test_packaging.py::test_bge_metadata_has_no_host_absolute_paths` and
the real `audit_package_bytes` runtime check both scan for this. If you must
sanitize it after a semantic store was already built against it: patch the
file, recompute `asset_tree_sha256`, patch the ONE `manifest.json` field that
records it, then re-run `build_semantic_store` (without `--force`) and
confirm it reports `status: "reused"` before trusting anything downstream —
that proves the embeddings themselves are untouched. Then delete only the
`semantic_fold_*.joblib` checkpoints (not `sparse_fold_*`) in every affected
run directory before re-running CV, or you'll hit
`ValueError: invalid semantic checkpoint` (fail-closed, working as intended
— it correctly detected the manifest changed).

## Quick orientation for common tasks

- Full clean-room rebuild pattern (re-embed → sparse CV → semantic CV →
  ensemble eval → robust CV → train_final → double-build zip): see any
  `build_v0*.sh` / `resume_v0*_from_stage*.sh` script at the project root
  for the exact stage sequence and flags.
- Graph-feature (v08) tooling: `scripts/build_graph_features.py` (extraction,
  ~35 min, CPU-only, safe to run alongside anything) and
  `scripts/graph_ablation_cv.py` (CV comparison, needs the same RAM headroom
  as a semantic CV stage — don't run concurrently with another heavy job).
- Session-disjoint CV protocol (`session_grouped_folds` in `validation.py`)
  is wired into `scripts/run_robust_cv.py` as an additional, separately
  reported `session_protocol` field — it is NOT part of the strict
  objective-disjoint promotion gate (different generalization axis), so
  don't fold it into gate-pass/fail logic without thinking about why.
