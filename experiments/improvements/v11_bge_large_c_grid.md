# v11 — bge-large regularization hypothesis  ❌ REFUTED

**Submission:** none — settled by a semantic-only grid search before any
robust-CV/full-pipeline work.

## What this was testing

`docs/agent_handoff.md`'s v07 section explicitly flags an untested,
plausible explanation for why bge-base → bge-large regressed local CV
broadly: *"semantic_c=0.1 ... never retuned for the larger embedding
space."* This tested that specific, narrow hypothesis — not "does scaling
the encoder help" (already answered no, by v07 itself) — with real data,
which hadn't been possible before this session (v07 was originally built on
a different host, before this session's raw data existed here).

## Method

Standalone script (same pattern as the H-B kill-switch checks): encoded all
35,072 responses + 398 objectives with `bge-large-en-v1.5` directly via
`load_sentence_transformer`/`encode_texts` (config.py never touched, so
none of the "config.py points at the wrong candidate" coordination risk
CLAUDE.md warns about). Config.json's leaked host path
(`/root/.cache/torch/...`, already documented) was sanitized the same way
as bge-base's. Grid-searched `semantic_c` over
`(0.01, 0.03, 0.05, 0.1, 0.2, 0.5)`, refitting `fit_semantic_model` on the
same objective-disjoint folds/seed as production for each value.

## Result

| C | primary log loss | roc_auc |
|---|---|---|
| 0.01 | 0.59889 | 0.5886 |
| 0.03 | 0.59602 | 0.5978 |
| **0.05** | **0.59589 (best)** | 0.5997 |
| 0.1 (v07's original value) | 0.59738 | 0.5997 |
| 0.2 | 0.60066 | 0.5977 |
| 0.5 | 0.60705 | 0.5946 |

Reference: bge-base semantic-only (production `C=0.1`) = **0.59177**.

**Every single C value tested is worse than bge-base's semantic-only
result** — including the best one (0.05), which is still +0.00412 worse
than bge-base, well above this project's noise floor (~0.0015-0.002).

## Conclusion

**The regularization hypothesis is refuted, not just untested anymore.**
Retuning `semantic_c` narrows the gap (0.1 → 0.05 improves bge-large by
~0.0015) but does not close it, let alone reverse it. bge-large is
genuinely a worse standalone semantic encoder than bge-base for this
specific tutoring-dialogue relevance-scoring task, across the whole
regularization range that would plausibly matter — this isn't an
under-regularization artifact of v07's original single-C attempt.

This resolves the open question the docs flagged, so a future session
doesn't need to re-ask it: **bge-large is closed as a lever, confirmed by a
full C sweep on real data, not just the original single-C v07 result.**
Building the full 4-head-equivalent pipeline (robust CV, train_final,
packaging) for bge-large was correctly skipped — this semantic-only sweep
already answers the question a full build would have, at a fraction of the
cost.
