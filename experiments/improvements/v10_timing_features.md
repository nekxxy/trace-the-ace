# v10 — per-turn timing/latency dense features  ❌ NOT ADOPTED

**Submission:** none — ablation only, never built for submission.
**Module:** `src/trace_ace/timing_features.py` (9 features), extracted via
`scripts/build_timing_features.py` to
`data/interim/timing_features/timing_dense.parquet`.

## What this was testing

Existing dense features (`duration_seconds`, `utterances_per_minute` in
`features.py`) only capture whole-session totals. This tested whether
*local, per-turn* pacing carries independent signal: how long a student
takes to respond to a specific tutor turn, whether that latency trends up
or down over a session, whether it changes specifically around hints vs
corrections, and overall pacing consistency — none of which a whole-session
average/rate can express, and none of which is a transform of the cached
bge embeddings (unlike v08's graph features), so this was a genuinely
different signal source in that sense.

Nine features: mean/std/trend of student response latency, longest gap in
the session, mean latency specifically after hint-tagged vs
correction-tagged tutor turns, coefficient of variation of all consecutive
gaps, and fast-response/slow-response ratios (thresholds 3s / 30s). All are
pure functions of one transcript (timestamps are HH:MM:SS, real
second-level resolution, confirmed on raw data), no label input — same
leakage-safety pattern as `graph_features.py` (see
`test_signature_has_no_label_shaped_input`).

## Ablation protocol

Same method as v08: appended to the existing v06 bge-base semantic store's
dense block, re-fit with the exact same `fit_semantic_model`/folds/config as
production, primary (objective-disjoint) and session-disjoint protocols.
Reused `scripts/graph_ablation_cv.py` unchanged — it's generic over any
`row_id`-keyed dense-feature parquet, so no new ablation script was needed.

## Result

| protocol | baseline | with timing features | delta |
|---|---|---|---|
| primary (objective-disjoint) | 0.59177 | 0.59185 | **+0.00008 (worse)** |
| session-disjoint | 0.55714 | 0.55721 | **+0.00007 (worse)** |

Both deltas point the *wrong* direction (worse, not better), though the
magnitude is tiny — about 10x smaller than v08's already-below-floor
graph-feature lift, and consistent with pure noise rather than a real
effect either way. **Not adopted.**

## Interpretation

Unlike v08 (real-direction but too small) or the three H-B candidates
(real diversity but zero blend value), this one doesn't even show a
believable positive direction — it's flat-to-negative noise. Plausible
reading: the tutoring-dialogue pacing signal that would matter (confusion,
disengagement, confident quick answers) is likely already substantially
captured by the existing `tutor_feedback_count`/`student_uncertainty_ratio`-
style early/mid/late dense features and by the semantic content of the
responses themselves (BGE embeddings of what a student *actually wrote*
probably already correlate with how long they took to write it). Timing
alone, independent of content, doesn't appear to add exploitable signal
for this specific task.

**Don't re-try this exact feature set expecting a different answer.** If
transcript timing is revisited, it would need a genuinely different framing
than "more per-turn latency statistics" — e.g. something conditioned on
content (was the response fast *and* wrong, vs slow *and* wrong), which
starts to overlap with feature-interaction territory rather than a new
independent signal.
