# v09 (H-B) — all-MiniLM-L6-v2 as an additive 4th head  ❌ KILLED BEFORE BUILD

**Submission:** none — killed by the correlation gate before any ensemble architecture work.
**Candidate asset:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, MiniLM distillation, 384-d), registered in `configs/resources.json` as `all_minilm_l6_v2`.

## What this was testing

`docs/agent_handoff.md`'s v08 (graph-features) section identifies "H-B" — a second,
independently-pretrained small embedding model added as an **additive 4th ensemble
head** alongside bge-base, with a mandatory correlation kill-switch before investing
in the full build — as the most promising untried lever, since graph features (v08)
gave real-direction but 10-15x-below-noise-floor lift, and scaling the encoder
further (bge-large, v07) regressed local CV broadly.

`all-MiniLM-L6-v2` was chosen because it's already referenced as a considered
candidate in `docs/history/PROJECT_LEARNING_LOG.md`, and it's a genuinely different
lineage from BAAI/bge: MiniLM distillation architecture, trained by the
sentence-transformers team on a different corpus/objective mix (contrastive
training over 1B+ sentence pairs from many sources) rather than BGE's own
pretraining recipe.

## Kill-switch protocol

Reused the exact same core functions as production (`fit_semantic_model`,
`build_semantic_interaction_matrix`, `objective_disjoint_folds`, same seed
`20260716`, same `semantic_c=0.1`) via a one-off script rather than the full
store-building pipeline (which hardcodes `BGE_DIMENSION=768` and would have
required touching `config.py` — the exact class of mistake `CLAUDE.md`
repeatedly warns against). Encoded all 35,072 response contexts + 398
objectives with MiniLM, fit a semantic-only model per objective-disjoint fold,
and correlated the resulting OOF probabilities against the existing bge-base
semantic OOF (`experiments/runs/cleanroom_v06/semantic_oof.parquet`).

## Result

- MiniLM-alone semantic OOF: log loss **0.59813**, AUROC 0.60325 — actually
  *worse* than bge-base's own semantic-only OOF (0.59177), consistent with
  MiniLM being a smaller, more generic, older-generation encoder than BGE.
- **`corr(bge_base_sem_oof, minilm_sem_oof) = 0.8759`** — above the documented
  0.85 kill-switch threshold.

**Killed per protocol.** No diversity gain to justify building the additive
4th-head ensemble architecture (which would have meant real changes across
`config.py`, `ensemble.py`, `training.py`, `validation.py`,
`evaluate_ensemble.py`, `run_robust_cv.py`, `train_final.py`,
`build_submission.py`, `verify_submission_runtime.py`, and
`submission_template/main.py`) — that work was correctly not started.

## What this means for H-B going forward

This doesn't close the H-B lever entirely, just this specific candidate: a
0.88 correlation says MiniLM and BGE learn substantially overlapping semantic
signal on this domain (both are generic sentence-embedding models trained
with broadly similar contrastive objectives), not that *no* second encoder
could add diversity. A genuinely different embedding paradigm — e.g. a
model with meaningfully different training data/objective (not just a
different architecture on similar training recipes) — would be a more
promising next candidate than trying another generic sentence-transformer
encoder and expecting a different correlation result.
