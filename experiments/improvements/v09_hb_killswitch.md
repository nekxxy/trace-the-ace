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

## Second attempt (2026-07-27): mean-pooled DistilBERT — passes the correlation kill-switch, still not adopted

Tried `distilbert-base-uncased` (Apache-2.0), mean-pooled via
`sentence_transformers.models.Transformer` + `models.Pooling(mode="mean")`
rather than loaded as a pre-packaged sentence-transformers model. Chosen
specifically to isolate the training-objective variable: same general
BERT-family encoder architecture as bge-base, but MLM+distillation only,
**no contrastive similarity fine-tuning at all** — the classic SBERT-paper
"vanilla BERT mean pooling" baseline.

- DistilBERT-alone semantic OOF: log loss **0.60025**, AUROC 0.58792 — worse
  than both bge-base alone (0.59177) and MiniLM alone (0.59813), as expected
  for an encoder with no similarity fine-tuning.
- **`corr(bge_base_sem_oof, distilbert_sem_oof) = 0.8337`** — below the 0.85
  threshold. **Passes the kill-switch** (also only 0.760 correlated with
  MiniLM's OOF, confirming this really is a different signal from both).

Before building the actual 4th-head ensemble architecture (real changes
across `config.py`, `ensemble.py`, `training.py`, `validation.py`,
`evaluate_ensemble.py`, `run_robust_cv.py`, `train_final.py`,
`build_submission.py`, `verify_submission_runtime.py`,
`submission_template/main.py` — not a small change), ran one more cheap
check first: a grid search (step 0.025) over all four blend weights
(full/role/bge-semantic/distilbert, nonnegative, summing to 1, matching
`blend_probabilities`'s existing linear-blend family) using the *already
computed* OOF from `experiments/runs/cleanroom_v06/ensemble_oof.parquet` and
this candidate's OOF — no new model architecture needed to answer "would
this actually help."

**Every one of the top 10 weight combinations by OOF log loss assigns
DistilBERT a weight of exactly 0.000.** The best 4-way blend (0.58742) is
statistically indistinguishable from the current 3-way blend (0.58744) —
both within the same ±0.00004 noise band as just re-tuning the existing
full/role/semantic weights slightly. **Not adopted** — the ensemble
architecture was correctly never built, since this grid search already
proves it wouldn't have helped.

**The generalizable lesson:** the correlation kill-switch (`< 0.85`) is
necessary but *not sufficient*. A candidate can be genuinely diverse (low
correlation with the existing head) while still carrying too little own
predictive signal to be worth the noise it adds to a blend — diversity and
value are different properties, and both need checking before investing in
the full architecture. A future H-B candidate should clear *both* bars:
correlation below 0.85 **and** a nonzero optimal weight in a cheap
OOF-blend grid search using the existing production OOF files, before any
ensemble architecture work starts.

## Third attempt (2026-07-27): pure-NLI mpnet — passes the kill-switch, still zero blend value

Tried `sentence-transformers/nli-mpnet-base-v2` (Apache-2.0, mpnet-base,
768-d) — fine-tuned **only** on SNLI+MultiNLI (entailment/contradiction/
neutral classification), no retrieval-contrastive or STS-specific training
at all. A third, distinct point in the training-objective space from the
first two candidates.

- NLI-mpnet-alone semantic OOF: log loss **0.60629**, AUROC 0.58324 — the
  *weakest* of all three candidates tried (bge-base 0.59177, MiniLM 0.59813,
  DistilBERT 0.60025, NLI-mpnet 0.60629).
- **`corr(bge_base_sem_oof, nlimpnet_sem_oof) = 0.8108`** — the *most*
  decorrelated of the three. **Passes the kill-switch.**
- Same cheap 4-way blend-weight grid search as before: **every one of the
  top 10 weight combinations assigns NLI-mpnet a weight of exactly 0.000.**
  Best 4-way blend (0.58742) is, again, noise-floor-indistinguishable from
  the current 3-way blend (0.58744). **Not adopted.**

Note: `nli-mpnet-base-v2`'s `config.json` also leaks a path
(`old_models/nli-mpnet-base-v2/0_Transformer` — a relative path, not the
host-absolute pattern `audit_package_bytes` scans for) — not sanitized
since this candidate was killed before reaching any packaging step.

## Conclusion after three candidates: this specific lever (off-the-shelf pretrained encoder swapped in as a 4th head) looks structurally exhausted, not just unlucky

Across three methodologically distinct candidates spanning the training-
objective spectrum, a clear, consistent pattern emerged:

| candidate | training objective | corr vs bge-base | standalone log loss | optimal blend weight |
|---|---|---|---|---|
| bge-base (existing) | retrieval-contrastive | — | 0.59177 | (production) |
| all-MiniLM-L6-v2 | retrieval-contrastive (different corpus) | 0.876 (fails) | 0.59813 | n/a - killed pre-blend-check |
| distilbert-base-uncased | none (MLM+distillation only) | 0.834 (passes) | 0.60025 | 0.000 |
| nli-mpnet-base-v2 | NLI classification only | 0.811 (passes) | 0.60629 | 0.000 |

Correlation and standalone quality move together: encoders trained further
from bge-base's own objective decorrelate *and* get worse at this specific
task simultaneously. That's not three independent unlucky draws — it's
consistent with bge-base's retrieval-contrastive training being closely
matched to the kind of relatedness that predicts this label, so moving
away from that training recipe in *any* direction trades away standalone
quality faster than it buys usable diversity. **Recommend not trying a
fourth off-the-shelf pretrained encoder in this vein expecting a different
answer** — that would be noise-mining, not principled search, given the
pattern is now consistent across three well-chosen, distinct candidates.
A genuinely different next step would need either a much larger/more
capable second encoder (reintroducing the "closed lever" capacity-scaling
caution from v07), or a feature source that isn't a pretrained text
encoder at all (e.g. transcript timing/latency signal, unexplored so far).
