# Agent handoff — 2026-07-24 to 2026-07-25 session

This is a detailed narrative of one work session, kept separate from
`experiments/improvements/README.md` (which stays a concise scoreboard) so a
future agent can understand *why* things are the way they are without
re-deriving it. `CLAUDE.md` at the project root is the short version; this is
the long version.

## Starting point

A prior session had left `resume_v06_from_stage3.sh` mid-flight: Stage 1
(bge-base semantic store rebuild) and Stage 2 (sparse CV) had completed, but
Stage 3 (semantic CV) had been OOM-killed at 2026-07-23 20:33 UTC — the host
has ~3.8GB RAM / 2 cores / no GPU, and `train_semantic_cv` briefly needs
~1.8GB+ peak per fold on a 768-dim interaction matrix (StandardScaler copy +
transformed copy + liblinear's internal copy, on top of the full held
interaction matrix). The script deliberately sets its own
`oom_score_adj=800` so it's the sacrificial victim under system-wide memory
pressure rather than some other service on the box — that's why it was the
one killed, not a sign anything else was wrong.

## v06: added a 4GB swap file, resumed, it worked, and it was real

Added `/swapfile_v06` (4GB, `swapon`), resumed from Stage 3 (skipping the
already-good Stage 1/2 output), and the full pipeline completed in ~33
minutes: primary ensemble log loss **0.58742** local, robust CV clean across
all 4 semantic-family protocols, deterministic double-build zip
(`trace_ace_cleanroom_v06.zip`, sha256 `e55f1c7e...`, 261,706,901 bytes).

The user submitted it and reported back the real score: **0.6073 log loss,
0.6248 AUROC, rank 20**. This mattered a lot, because the project's own
`experiments/improvements/README.md` had a **2026-07-21 decision to lock v03
as final**, with a documented rationale: local CV tracks real score via a
~+0.02 "transfer gap" for honest changes, but v05 (an encoder swap to
gte-small) had sharpened the local fit at the cost of a wider, worse-live
gap — so the standing conclusion was "we're at the non-overfit floor, going
lower is an overfitting trap." v06's real result **broke that pattern in the
good direction**: local 0.58742 → real 0.6073, gap = +0.0199, actually the
*tightest* gap of any candidate (v02/v03's honest gap was +0.0203-0.0205).
That's not luck — it's a real signal that a genuine capacity upgrade in the
*same encoder family* (bge-small → bge-base, same architecture, just bigger)
transfers differently than a *lateral* swap to a different, smaller family
(v05's gte-small). The lock was explicitly superseded (user decision,
2026-07-24) and the ledger updated with the revised gap-model reasoning.

Two pieces of hygiene got fixed at the same time, both cheap and safe to do
before/alongside the next candidate:

1. `_nested_full_calibration` in both `scripts/evaluate_ensemble.py` and
   `scripts/run_robust_cv.py` was fitting the Platt calibrator on the union
   of *other folds' validation indices*, which is **not** the same as a
   fold's own session-purged training rows — it silently re-included 13-19%
   of rows that share a session with the held-out fold. Fixed to use
   `fold.train_indices` directly. Impact on the existing 2-parameter Platt
   scaler is almost certainly sub-4th-decimal, but it was a real violation of
   the module's own purge guarantee and would matter for any future,
   higher-capacity recalibrator.
2. Added a 6th, session-disjoint CV protocol
   (`validation.session_grouped_folds`, which existed but was unused) to
   `run_robust_cv.py`, reported as a separate `session_protocol` field. This
   is a *different* generalization axis (unseen sessions, not unseen
   objectives) than the private test's actual objective-disjoint setup, so
   it's deliberately **not** folded into the strict promotion gate — it
   exists to catch a candidate that looks clean on every objective-disjoint
   split but is secretly exploiting session-level structure.

## v07: the next size up (bge-large) — built successfully, but it's a regression

Given v06's result, the obvious next experiment was pushing the same lever
further: bge-base → **bge-large-en-v1.5** (1024-dim, 24 layers vs bge-base's
12). Fetched the asset, swapped `config.py`'s `BGE_REPOSITORY`/`REVISION`/
`DIMENSION`, and ran the full clean-room pipeline from scratch (`build_v07.sh`).
Stage 1 (re-embed all 35,072 responses + 398 objectives) took ~13 hours —
roughly the ~3.5x compute-per-token estimate for 24 layers/1024-dim vs 12
layers/768-dim.

**Result: broad regression vs v06, not an improvement.**

| protocol | v06 | v07 | Δ |
|---|---|---|---|
| primary (objective-disjoint) | 0.58742 | 0.59067 | +0.00325 (worse) |
| semantic_k25_s0 | 0.59466 | 0.59931 | +0.00465 (worse) |
| semantic_k50_s0 | 0.58584 | 0.59708 | **+0.01124 (worse, large)** |
| semantic_k50_s1 | 0.59065 | 0.58623 | −0.00442 (better) |
| semantic_k80_s0 | 0.58925 | 0.59634 | +0.00709 (worse) |

3 of 4 robust protocols got meaningfully worse; only one improved. This is
not noise — it directly refutes the "same-family scaling keeps helping"
generalization from v06 (see the updated hypothesis note in the scoreboard).
Plausible explanations, untested: `semantic_c=0.1` and the 256-token
`bge_max_seq_length` truncation were never retuned for the larger embedding
space; bge-large may simply not suit this short, noisy tutoring-dialogue
domain as well past a certain capacity. **Do not resubmit v07, and don't
treat "keep scaling the encoder" as a proven-safe lever anymore** — it's
closed for now, not open-ended.

The zip was still built to completion (not aborted mid-way) — partly because
the compute was already sunk, partly to get a full, honest robust-CV picture
rather than judging off the primary number alone (this project's own
"noise floor" analysis explicitly warns against that), and partly because it
exercised the packaging pipeline for a meaningfully larger encoder, which
surfaced two real bugs described below.

### Bug 1: hardcoded `bge-base-en-v1.5` paths outside `config.py`

`scripts/build_submission.py` crashed at the zip-build step with
`RuntimeError: artifact and packaged BGE asset trees differ`. Root cause:
several files hardcode the *active* asset's directory name/repo id instead of
deriving it from `config.py`, and editing `config.py` alone (as done for the
bge-base → bge-large swap) missed all of them. This is the exact thing an
earlier internal research push had warned about ("must coordinate edits
across config.py, verify_submission_runtime.py REQUIRED_MEMBERS,
audit_integrity.py..."), and it's a hard, recurring trap — see the full file
list and grep command in `CLAUDE.md`. Fixed by mirroring the exact same
bge-small → bge-base diff that a prior session had already applied once
(found via `git diff` — the pattern was: edit the hardcoded path/key in each
of those files, plus their matching test fixtures). Ran the full test suite
(108 tests) after the fix; all pass.

### Bug 2: upstream `bge-large-en-v1.5/config.json` leaks a host path

After fixing bug 1, `tests/test_packaging.py::test_bge_metadata_has_no_host_absolute_paths`
failed: bge-large's `config.json`, **as published by BAAI on HuggingFace**,
has `"_name_or_path": "/root/.cache/torch/sentence_transformers/BAAI_bge-large-en/"`
baked in — a leftover from whoever originally converted/uploaded the
checkpoint. (bge-base's equivalent file is clean:
`"BAAI/bge-base-en-v1.5"`.) This isn't cosmetic-only: `verify_submission_runtime.py`'s
`audit_package_bytes` scans every packaged file for host-path byte markers
including `b"/root/"`, so this would very likely have blocked real runtime
verification, not just failed a local test.

The fix is more delicate than it looks because this project deliberately
hashes the *entire* asset directory byte-for-byte (`asset_tree_sha256`) at
every stage and fails closed if anything changes — by the time this was
caught, Stage 1's ~13-hour re-embed had already run against the *unsanitized*
file and baked that tree hash into the semantic store's manifest. The
resolution (confirmed safe, not a shortcut):

1. Patch `_name_or_path` in `assets/bge-large-en-v1.5/config.json` to the
   clean repo id.
2. Recompute `asset_tree_sha256` honestly (it's a real, different hash now).
3. Update `configs/resources.json`'s `bge_large_en_v1_5.asset_tree_sha256` to
   match.
4. Patch the **one** field in the semantic store's `manifest.json`
   (`source.asset_tree_sha256`) to the new value — nothing else in that file.
5. Re-run `build_semantic_store()` **without** `--force` and confirm it
   prints `"semantic store: unchanged inputs; existing embeddings are
   valid"` / returns `status: "reused"`. This is the load-bearing proof step:
   `_semantic_files_valid` independently re-verifies the actual embedding
   array file hashes against the manifest's `outputs` section, which is
   untouched by the config.json edit — so a "reused" result is a genuine,
   automatic confirmation that the embeddings themselves are unaffected, not
   an assumption.
6. Because the manifest *file's own bytes* changed (even though the arrays
   didn't), every downstream stage that recorded
   `semantic_store_manifest_sha256 = file_sha256(manifest.json)` is now
   stale relative to it — Stage 3 (semantic CV), Stage 4 (ensemble eval),
   Stage 5 (robust CV), Stage 6 (train_final) all needed to re-run. This
   reuses the untouched embedding arrays and the untouched sparse-CV
   checkpoints (sparse config never changed), so it's a ~25-40 minute
   re-run, not another 13-hour re-embed.
7. One more snag on the first re-run attempt: the *old* semantic fold
   checkpoints (`semantic_fold_*.joblib`, both in `cleanroom_v07/checkpoints/`
   and every `robust_cleanroom_v07/<protocol>/checkpoints/` directory) were
   still on disk from the pre-fix run and are keyed to the old manifest hash.
   `train_semantic_cv`'s checkpoint validation correctly detected the
   mismatch and raised `ValueError: invalid semantic checkpoint` rather than
   silently refitting — this is the fail-closed design working as intended,
   not a new bug. Fix: delete only the `semantic_fold_*.joblib` files (keep
   `sparse_fold_*.joblib`, which are unaffected), then re-run.

After all of the above, the corrected pipeline reproduced **byte-identical**
CV numbers to the pre-fix run (confirming the embeddings really were
untouched the whole time), and completed cleanly:
`trace_ace_cleanroom_v07.zip`, sha256
`7d6c3646f91c372b7e2c9a042f9cf6d39a96211abe18349f90541f7b849696ee`,
777,951,173 bytes, deterministic double-build. Verified the size is fully
explained by `model.safetensors` alone (1,340,616,616 of 1,353,414,849
uncompressed bytes, ~99%) — no stray files, same 27-entry manifest as v06's
zip, just a ~3x bigger encoder.

## v08: knowledge-graph-inspired features (in progress, not yet decided)

The user separately requested exploring "knowledge graph inspired" features
extracted per tutoring session (misconception, hint type, teaching strategy,
etc.), with an explicit ask to maximize leaderboard score and not worry about
elegance, and an explicit choice (after being shown the tradeoff) to include
leakage-adjacent feature names (`correction_success`, `reasoning_improvement_score`,
etc.) and to use heuristic/keyword extraction rather than an LLM pass.

Two things worth knowing if you pick this up:

1. **This is less risky than the feature names suggest**, once you look at
   how the existing pipeline already handles similar signal.
   `src/trace_ace/features.py`'s `_BASE_DENSE_FEATURE_NAMES` already ships
   (safely, in production since v02) things like `tutor_feedback_count`,
   `tutor_correction_count`, `student_uncertainty_ratio` at early/mid/late
   segments, and their late-minus-early deltas — i.e. "reasoning improvement"
   and "confidence change" style signal already exists and transfers fine.
   The competition's `is_correct` label is graded on a **separate follow-up
   question outside the visible transcript** (confirmed by an earlier
   internal leakage audit), so whole-session tutor/student text does not
   directly reveal the label being predicted for a given row — that's why
   `full_text`/`tutor_text`/`student_text` (the whole session, not just
   turns before "this" response) are already safely hashed into the `full`/
   `role` sparse heads in production. What *was* previously tried and
   rejected as unsafe was specifically **isolating the exact tutor-feedback
   verdict window as its own feature** (`student_answer_tutor_feedback_windows`
   as a dedicated hashed input) — described as "near-tautological." The new
   graph features are structural/behavioral counts (hint depth, strategy
   diversity, correction chains, action-transition graph density/degree),
   not a re-encoding of the verdict text itself, so they're a different and
   safer shape of feature than the one that was rejected — but see point 2.
2. **Built, tested, and extracted; NOT yet validated for lift.**
   `src/trace_ace/graph_features.py` (14 new dense features, pure functions
   of one transcript + one objective, no label input by construction — the
   function signature literally cannot see `is_correct`, see
   `test_signature_has_no_label_shaped_input` in
   `tests/test_graph_features.py`). Extracted for all 35,072 rows into
   `data/interim/graph_features/graph_dense.parquet` via
   `scripts/build_graph_features.py` (~32 min, CPU-only). Fixed one real bug
   during this work: `graph_density`/`graph_max_node_degree` initially
   exceeded their intended `[0,1]`/`[0, node_count-1]` bounds because
   same-type consecutive turns (e.g. two tutor hints in a row) were counted
   as self-loop edges; fixed by excluding `left == right` transitions, with a
   regression test (`test_graph_density_and_degree_are_bounded`).

   **Ablation run (2026-07-25), result: not adopted.**
   `scripts/graph_ablation_cv.py` appends the graph features to the existing
   v06 bge-base semantic store's dense block and compares primary +
   session-disjoint CV log loss with vs without, using the same
   `fit_semantic_model`/folds/metrics as production. This was deliberately
   held until v07 was fully done (it needs the same RAM class that required
   the swap file for a normal semantic-CV stage). Results:

   | protocol | baseline | with graph features | delta |
   |---|---|---|---|
   | primary (objective-disjoint) | 0.59177 | 0.59162 | −0.00016 |
   | session-disjoint | 0.55714 | 0.55704 | −0.00010 |

   Both deltas are real-direction (consistent improvement) but 10-15x below
   the ~0.0015-0.002 believable-effect-size floor this project established.
   This matches the original research push's own prediction for this exact
   class of feature ("GBM-sem already captured most of the exploitable
   nonlinearity in this matrix; near-free probes, expect tiny lift"). **Not
   adopted** — not wired into `feature_store.py`, no v08 submission built.
   Raw numbers are also saved at `experiments/runs/v08_graph_ablation.json`.

   One process note: this ablation run first crashed with
   `ValueError: semantic_feature_count must identify a non-empty feature
   prefix` because `config.py` was still pointed at bge-large (1024-dim, left
   over from v07) while the ablation loads v06's bge-base store (768-dim).
   Config was reverted back to bge-base (and every hardcoded-path file from
   the "when you change the BGE encoder" list in `CLAUDE.md` was reverted
   too, in the same coordinated fashion needed going the other direction) —
   worth remembering that abandoning a candidate needs the same care as
   adopting one.

   If graph-style features get revisited later, the lesson is: this specific
   lever (transforms of the same cached bge embeddings) is exhausted. A
   genuinely new signal source — most plausibly H-B from the original
   synthesis (a second, independently-pretrained small embedding model as a
   4th head, with the mandatory correlation kill-switch) — is the more
   promising remaining direction.

## Misc state you should know about

- A 4GB swap file (`/swapfile_v06`) is active on the host. It was added
  during v06's recovery and left in place since v07 also needed it; there's
  no strong reason to remove it unless disk space becomes tight (63GB free
  as of this session).
- `docs/README.md`'s documentation map now points here; `CLAUDE.md` at the
  project root is the short-form entry point for a new session.
