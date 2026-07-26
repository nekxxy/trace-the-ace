# Trace-the-Ace — Improve on the v3 candidate

You are continuing work in `/opt/trace-the-ace`. The current best submission is the **calibrated clean-room ensemble ("v3")**: three heads — `full` and `role` (sparse hashed-text logistic) + `semantic` (linear head on a bge-small-en-v1.5 interaction matrix) — blended `0.25/0.25/0.50`, with a Platt calibration applied to the over-scaled `full` head before the blend. Your goal: find a change that **improves the real leaderboard score beyond v3** and moves the rank up.

## Ground truth you MUST internalize before optimizing
- Metric: **log loss, lower is better.**
- v3 status: local objective-disjoint OOF ≈ **0.589**; **real public leaderboard ≈ 0.6094** (current best, and the number to beat).
- **CRITICAL — local CV is not a reliable proxy for the real score.** Established from prior validated experiments: real ≈ local **+ ~0.020** (a near-constant transfer gap), and **local OOF gains within ~±0.002 (the noise floor) do NOT reliably transfer.** A change that improved local by ~0.002 was confirmed to score *worse* on the real leaderboard (its transfer gap widened — i.e. it over-fit the local validation). Treat any local-only gain near the noise floor as unproven.
- Consequence: only trust a change that is **robust / generalizing by construction** (the calibration already in v3 is exactly this — it transferred cleanly), OR that clears a **large** local margin (well above ~0.002) across *all* validation protocols simultaneously.

## Already shown NOT to help — do not spend effort re-deriving these
- The **meta / blend / calibration recombination space is exhausted.** v3's single-component calibration + fixed 0.25/0.25/0.50 blend is the optimum there. Re-optimizing the blend weights **over-fits** (better local, worse on the robustness protocols).
- **Swapping in a different small (384-d) sentence-embedding model** improves local but does **not** transfer to the real board.
- **Nonlinear heads (gradient boosting, etc.) on the cached matrices** yield only marginal, non-clean local gains and add dependency/complexity.
- A mild **robustness recalibration** (temperature / shrinkage of the final blend toward the base rate) produces only a sub-noise-floor local change — not worth a scarce submission on its own.

## The one lever with a genuinely higher ceiling
A **stronger embedding class** for the semantic head — a base-size (768-d) or better sentence encoder — is the only avenue with real upside, because the semantic head is the workhorse and the only orthogonal signal in the ensemble. But:
- It **requires a GPU / higher-RAM machine.** The dev box has no GPU and OOMs on base-size models (only ~1.5 GB free RAM), so this cannot be built or validated on the current host.
- Transfer is **not guaranteed** — small-model swaps already failed to transfer. Validate any new encoder on the objective-disjoint **and** semantic-family-disjoint protocols, and require a clear, robust margin (not a noise-floor tick) that survives the ~0.02 local→real gap before trusting it.
- Packaging/runtime budget is ample (ZIP ~90 MB of a 60 GB limit; offline A100 runtime), so a larger asset is feasible to ship IF it validates.

## Hard rules
1. Do NOT weaken, skip, or relax any validation gate, integrity check, or provenance check. Achieve consistency by regenerating evidence, never by relaxing a check.
2. Validation is **objective-disjoint** ⇒ the private test has UNSEEN learning objectives ⇒ per-objective-ID priors/features do NOT generalize. Use only per-response/session features + objective-TEXT (semantic) signal. Each test row is scored independently — no cross-test-set aggregation.
3. Runtime must stay **offline** (`--network none`), deterministic, and batch-invariant to 1e-7. No competition data or transcript content leaves the machine.
4. **Preserve v3** as the known-good baseline (its model artifact + ZIP + backups). Any new candidate goes to new paths; never overwrite v3.
5. **Submission discipline:** submissions are limited and tied to a single account — never circumvent the limit (multi-accounting is disqualifying). Do NOT spend a scored submission on a sub-noise-floor local gain. Submit only a change that is robust by construction or clears a large local margin, and keep the best-scoring submission selected as the final entry.

## Suggested approach
1. Reproduce v3's local + runtime validation to confirm the baseline is intact.
2. Only if a GPU / large-RAM machine is available: build a stronger-encoder semantic head, validate across all protocols, and require a robust, well-above-noise margin before considering a submission.
3. Otherwise, treat v3 as at (or very near) the achievable ceiling from local work, and base any submission decision on robustness — not on chasing the local metric.

## Deliverables
Produce a new candidate ONLY if it clears the bar above, with: a full regenerated evidence chain, a deterministic ZIP + SHA-256, local extracted-runtime + official-container verification, and an honest local-score-vs-expected-real-score assessment. Otherwise, deliver a documented conclusion that v3 stands as the best candidate.
