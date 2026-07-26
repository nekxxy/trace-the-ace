# v05 — swap semantic encoder bge-small → gte-small  ❌ DID NOT TRANSFER

**Submission:** id-2135 · **Real public score: 0.6105** (WORSE than v03's 0.6094) · Local OOF: 0.58712
**Artifact:** models/final_ensemble_cleanroom_v05.joblib · ZIP sha256 0ccc01d5…

## What changed vs v03
Replaced the semantic embedding model from `bge-small-en-v1.5` with `thenlper/gte-small` (both 384-d, MIT). Everything else identical — same calibration, same 0.25/0.25/0.50 blend. gte was a slightly better standalone encoder locally.

## What happened
- **Local:** 0.58914 → 0.58712 (−0.00202), and on *matched folds* it improved all 5 protocols. Batch-invariant, deterministic; passed local + container + platform Smoke.
- **Real:** 0.6094 → **0.6105 (+0.0011 — WORSE)**. The local gain reversed on the real test.

## Why it did NOT transfer (the lesson)
- gte's improvement was a **sharp fit to the local validation's objective set**, not a robust gain. Its local↔real gap (0.6105 − 0.58712 = 0.023) was *wider* than v03's (0.6094 − 0.58914 = 0.020) — a classic over-fitting tell.
- The −0.002 local gain sat right at the noise floor (~0.0015–0.002), where local CV cannot distinguish real signal from validation-specific fit.
- **Takeaway:** do not swap a core component for a small local gain and expect it to hold on the real test. Encoder swaps are high-variance; the reliable wins are robust corrections (see v03).

Not the submission to select. v03 (0.6094) is better on the real leaderboard.
