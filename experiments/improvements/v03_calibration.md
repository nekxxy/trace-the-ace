# v03 — Platt-calibrate the `full` head  ✅ WINNER (current best)

**Submission:** id-2027 · **Real public score: 0.6094** · Local OOF: 0.58912
**Artifact:** models/final_ensemble_cleanroom_v03.joblib · ZIP sha256 9c0d24ac…
**Link:** https://sera.64-227-166-78.sslip.io/static/downloads/submission2_trace_ace_calibrated_9c0d24ac.zip

## What changed vs v02
The `full` sparse head is over-scaled (learning_rate='optimal', alpha 3e-5) → its logits are too confident. Fit a 2-parameter Platt scaler `sigmoid(a·logit(full)+b)` on out-of-fold predictions, apply it before the unchanged 0.25/0.25/0.50 blend. Nothing else changed.

## Why it improved the score
- **Local:** 0.59149 → 0.58912 (−0.00237). Cuts ECE 0.0346 → 0.0172, improves worst fold, improves all 4 robustness protocols.
- **Real:** 0.6120 → **0.6094 (−0.0026)** — the local gain transferred almost exactly.

## Why it TRANSFERRED (the lesson)
It's a **robust, low-capacity correction** (2 parameters fixing a known miscalibration), not a fit to local quirks. This is the kind of change that generalizes from local validation to the real test. It remains the best submission.
