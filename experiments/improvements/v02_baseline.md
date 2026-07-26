# v02 — clean-room baseline

**Submission:** id-1993 · **Real public score: 0.6120** · Local OOF: 0.59149

## What it is
The clean-room ensemble: three heads blended with fixed weights.
- `full`, `role` — sparse hashed-text logistic heads (SGD)
- `semantic` — linear head on a 1618-d interaction matrix from **bge-small-en-v1.5** embeddings
- Blend: `0.25·full + 0.25·role + 0.50·semantic`, clipped.

## Result
Baseline. Everything after is measured against this. Real test (0.6120) ran ~0.021 above local OOF (0.59149) — the first sign that the real distribution is harder/shifted vs local validation.
