# Trace-the-Ace — Improvement Log

Metric: **log loss on the held-out test set, lower is better.** One `.md` per candidate.

## The one lesson that matters
**Local cross-validation (objective-disjoint OOF) is NOT a reliable predictor of the real leaderboard score.** The public test is a *different, harder distribution* — real scores run ~0.02 higher than local OOF across every candidate. What we've learned about *which* local changes survive the jump:

- **Robust, low-capacity changes transfer.** The v03 calibration (a 2-parameter Platt scaler) improved local *and* real by ~the same amount.
- **Sharp fits to local structure do NOT transfer — they can reverse.** The v05 encoder swap (gte) improved local by −0.002 but scored **+0.001 worse** on the real test, and its local↔real gap was *wider* than v03's — a tell-tale sign of over-fitting the local validation.

Practical rule going forward: only believe a candidate if the change is *robust/generalizing* by construction, and treat any local-only gain near the noise floor (±0.002) as unproven until a real submission confirms it.

## Scoreboard (real public score = source of truth)
| version | change | local OOF | **real public** | real Δ vs prev best | transferred? | submission |
|---|---|---|---|---|---|---|
| v02 | clean-room baseline: 3 heads (full/role/semantic) blended 0.25/0.25/0.50 | 0.59149 | **0.6120** | — (baseline) | — | id-1993 |
| v03 | + Platt-calibrate the `full` head before the blend | 0.58912 | **0.6094** | **−0.0026 (better)** | ✅ yes | id-2027 |
| v04 | + LightGBM head on the semantic matrix (4-comp blend) | ~0.58809 | not submitted | — | — | never built |
| v05 | swap semantic encoder bge-small → gte-small (different family) | 0.58712 | **0.6105** | **+0.0011 (worse)** | ❌ no | id-2135 |
| **v06** | upgrade semantic encoder bge-small → **bge-base** (same family, higher capacity) | 0.58742 | **0.6073**, AUROC 0.6248, rank 20 | **−0.0021 (better)** | ✅ yes | (2026-07-24) ← **BEST** |
| v07 | upgrade semantic encoder bge-base → **bge-large** (same family, next size up) | 0.59067 | not submitted | — | ❌ regressed locally | never built for submission |
| v08 | + 14 knowledge-graph-inspired dense features (hint depth, strategy diversity, correction chains, action-graph density/degree) on the bge-base semantic matrix | semantic-only: 0.59162 (Δ −0.00016 vs 0.59177) | not submitted | — | ❌ below noise floor | never built for submission |
| v09 | H-B: three additive-4th-head candidates (MiniLM, DistilBERT, NLI-mpnet) — one killed by correlation, two killed by zero-weight blend check | MiniLM 0.59813; DistilBERT 0.60025; NLI-mpnet 0.60629; best any 4-way blend: 0.58742 (= noise vs current 0.58744) | not submitted | — | ❌ no ensemble value, lever looks exhausted | never built (none of the three candidates helped) |

**Current best submission: v06 = 0.6073 (rank 20 public).** ← supersedes the 2026-07-21 "v03 locked as final" decision (user decision, 2026-07-24): v06 is a real, confirmed improvement over v03, not noise.

**v07 built (2026-07-25), but NOT recommended for submission.** Robust CV against v06 across all 4 semantic-family protocols: worse on 3/4, sometimes substantially (k50_s0: 0.58584 → 0.59708, +0.0112). This breaks the "same-family capacity upgrade always transfers" working hypothesis below — the pattern held for bge-small→bge-base but reversed for bge-base→bge-large. Zip was still built end-to-end (`trace_ace_cleanroom_v07.zip`) as a complete, honest record and to validate the packaging pipeline for a larger encoder, not because it's a good candidate. Full artifact/hash details and the packaging bugs found along the way are in `docs/agent_handoff.md`.

Smoke-test rows on the platform (scores ~0.50–0.52) are run on a tiny fixture and are NOT comparable to full scored submissions — ignore them for ranking.

## Updated transfer-gap model (4 real submissions)
The real score tracks local OOF plus a near-constant "transfer gap":

| version | local OOF | real | gap (real − local) |
|---|---|---|---|
| v02 | 0.59149 | 0.6120 | +0.0205 |
| v03 | 0.58912 | 0.6094 | +0.0203 |
| v05 | 0.58712 | 0.6105 | +0.0234 (inflated — over-fit) |
| v06 | 0.58742 | 0.6073 | **+0.0199** (honest — tightest gap yet) |

- The "honest" gap band is **~+0.0199–0.0205** (v02, v03, v06 all land in this narrow range). v05 is the outlier at +0.0234, and it's the only change in this table that came from *recombining/re-tuning existing signal* rather than upgrading a component outright.
- **Revised lesson:** the earlier conclusion ("v03 is the ceiling, anything lower is an over-fit trap") was **too broad**. What actually failed to transfer was v05's *lateral* swap to a smaller, different-family encoder (gte-small) chasing a marginally better local fit. What *did* transfer — cleanly, with the tightest gap of any candidate — was v06's *vertical* upgrade to a strictly higher-capacity encoder in the **same family** (bge-small → bge-base, same architecture/tokenizer, just bigger).
- **Working hypothesis for v07 and beyond (PARTIALLY REFUTED, see below):** genuine capacity/signal upgrades (bigger/better encoders, new orthogonal features) tend to transfer; recombination/calibration/weight-tuning tricks on the same fixed signal (the "meta space") are the ones that risk over-fitting the local split without adding real signal. Still only 4 data points — treat this as a working hypothesis to keep testing, not a proven law, and still require every candidate to pass the "clean across all CV protocols" gate before submitting.

**v07 update (2026-07-25):** the "bigger same-family encoder keeps helping" hypothesis does NOT extend indefinitely — bge-base → bge-large regressed local CV broadly (3 of 4 robust protocols worse, not a rounding error). The bge-small → bge-base step was not a general rule ("bigger is better"); it may have been a local sweet spot, or bge-large may need something v06 didn't (different regularization for the larger embedding space, longer than the current 256-token truncation, etc. — untested). Treat "scale the encoder further" as a closed lever for now, not an open-ended one — don't re-try bge-large or go straight to bge-xlarge-class models on the strength of v06 alone.

**v08 update (2026-07-25):** tested 14 knowledge-graph-inspired dense features (hint depth/count, strategy diversity, correction-chain length, tutor-intervention count, concept-transition count, reasoning/confidence change, misconception persistence/unresolved count, action-transition graph density and node-degree stats — all pure functions of one transcript + its own objective, no label input) appended to the semantic head's existing dense block, on the already-built v06 bge-base store. Primary delta −0.00016, session-disjoint delta −0.00010 — both real-direction but 10-15x below the ~0.0015-0.002 believable-effect-size floor established for this project. Consistent with the earlier research push's prediction for this exact class of feature ("GBM-sem already captured most of the exploitable nonlinearity in this matrix; near-free probes, expect tiny lift"). **Not adopted.** The extraction module (`src/trace_ace/graph_features.py`, tested, leakage-safe by construction) and tooling (`scripts/build_graph_features.py`, `scripts/graph_ablation_cv.py`) are kept in the repo since they're reusable if a genuinely new signal source (not just new transforms of the same cached embeddings) is tried later — see the "feasible levers" ranking in `experiments/runs/top5_20260720_v5/context_pack.md` for what's still untried (H-B: a second, independently-pretrained small embedder as a 4th head, is the one lever with real ceiling that hasn't been attempted).

**v09 update (2026-07-27):** attempted H-B with `all-MiniLM-L6-v2` (Apache-2.0, MiniLM distillation, 384-d) as the second embedder. Ran the mandatory correlation kill-switch (fit a semantic-only model on MiniLM embeddings alone, same folds/seed as production, correlate its OOF against bge-base's) *before* building any ensemble architecture — `corr(bge_base_sem_oof, minilm_sem_oof) = 0.876`, above the 0.85 threshold, and MiniLM-alone log loss (0.59813) is worse than bge-base-alone (0.59177) besides. **Killed pre-build**, no additive-4th-head architecture work was done. This doesn't close the H-B lever entirely — it says generic sentence-transformer encoders trained with broadly similar contrastive objectives overlap too much with bge-base on this domain, not that no second encoder could add diversity.

**v09 update #2 (2026-07-27):** tried mean-pooled `distilbert-base-uncased` (no contrastive similarity fine-tuning at all, isolating the training-objective variable) as a second H-B candidate. This one **passed** the correlation kill-switch (`corr = 0.834`, also only 0.760 vs MiniLM's OOF — genuinely different signal from both), but a cheap grid search over all 4 blend weights using already-computed OOF (no new architecture needed to check this) found the optimal weight for DistilBERT is **exactly 0** — the best 4-way blend (0.58742) is noise-floor-indistinguishable from the current 3-way blend (0.58744). **Not adopted; ensemble architecture correctly never built.**

**v09 update #3 (2026-07-27):** tried `nli-mpnet-base-v2` (fine-tuned only on SNLI+MultiNLI entailment classification, no retrieval/STS training) as a third candidate. **Passed** the kill-switch (`corr = 0.811`, the most decorrelated yet), but hit the same zero-weight result in the blend grid search — not adopted. Across all three candidates, correlation and standalone quality moved together (more decorrelated → weaker alone: bge-base 0.59177, MiniLM 0.59813, DistilBERT 0.60025, NLI-mpnet 0.60629), and none contributed positive blend weight. **Conclusion: this lever (swapping in an off-the-shelf pretrained encoder as an additive 4th head) looks structurally exhausted after three methodologically distinct tries, not just unlucky — do not try a fourth generic encoder expecting a different answer.** Full protocol and all three results in `experiments/improvements/v09_hb_killswitch.md`.

