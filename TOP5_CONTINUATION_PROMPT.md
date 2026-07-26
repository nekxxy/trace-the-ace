# Trace the Ace Top-5 Continuation Prompt

You are continuing work in `/opt/trace-the-ace`. The user explicitly rejected uploading the current candidate as Normal and wants an aggressive, evidence-driven push toward a **top-5 leaderboard rank**.

Work autonomously, use as much safe parallelism as the environment supports, and coordinate specialist agents if the multi-agent tools are available. If a tool can spawn agents, start with 20-40 focused agents and scale upward only when their outputs remain useful; do not create 100 vague agents. The goal is not more activity, it is finding validated improvements that can plausibly beat the current clean-room candidate.

## Current Verified State

- Repo: `/opt/trace-the-ace`
- Branch: `recovery/prestores-20260718`
- HEAD: `3846dedb03caa1ad97620e0b6c2a287c7041beb5`
- Current local tests: `venv/bin/python -m pytest -q` -> 87 passed
- Current candidate ZIP:
  - Local path: `/opt/trace-the-ace/submissions/builds/trace_ace_cleanroom_v02.zip`
  - Public link: `https://sera.64-227-166-78.sslip.io/static/downloads/submission1_trace_ace_cleanroom_v02.zip`
  - SHA-256: `ff9578e9f3494018df996cd7fa3d4a60efd1d7ff0e32b0c385f297fc371651e0`
  - Size: `88,673,009` bytes
  - ZIP entries: `26`
- Current final model:
  - Path: `/opt/trace-the-ace/models/final_ensemble_cleanroom_v02.joblib`
  - SHA-256: `87ad03d488242fe5bfe91905251a73bc7010694f9be20abd0a43fa2bc6876738`
- Local extracted runtime: passed.
- Pinned official container runtime: passed twice.
- Platform Smoke evidence from user log:
  - File: `/opt/trace-the-ace/experiments/runs/cleanroom_v02/platform_smoke_user_log_20260719.md`
  - Log showed unpack success, root `main.py`, `python main.py` success, `submission.csv` created/copied, exit code `0`.
  - Caveat: pasted log did not include job ID or uploaded ZIP SHA-256. Treat as success for this candidate only if the uploaded file hash was the ZIP hash above.

Current candidate validation:

- Primary ensemble log loss: `0.5914910112291625`
- Primary ensemble AUROC: `0.6220338208730392`
- Primary Brier: `0.2017873332740961`
- Primary ECE10: `0.03464602307815701`
- Primary worst fold log loss: `0.6185917370921679`
- Robustness gate: passed.
- Worst robustness protocol: `semantic_k25_s0`

## Hard Rules

1. Do **not** upload a Normal submission without explicit user authorization.
2. Do **not** weaken validation gates to make a candidate look better.
3. Do **not** expose, print, transmit, commit, or package raw competition transcripts or test-derived content.
4. Do **not** use external APIs to process transcript text or competition data.
5. Do **not** edit `data/source`.
6. Do **not** use leaderboard feedback as a tuning oracle unless the rules explicitly allow the specific action and the user authorizes quota use.
7. Preserve the current known-good candidate before changing model code or artifacts.
8. Any new ZIP must get a public link named with `submissionN` keywords, such as `submission2_trace_ace_<short_name>.zip`, and the link must be reported with hash and size.
9. If authenticated platform access exists, Smoke submissions are allowed only for exact packaged candidates; Normal submissions still require explicit user confirmation.
10. Online use is allowed only for public docs, public code ideas, package documentation, and general ML research. Do not send competition data or transcript content online.

## First Actions

1. Read these files before acting:
   - `README.md`
   - `SUBMISSION_READINESS_PROMPT.md`
   - `docs/implementation_decisions.md`
   - `docs/competition/competition_rules.md`
   - `docs/competition/code_submission_format.md`
   - `experiments/runs/cleanroom_v02/final_handoff.json`
   - `experiments/runs/cleanroom_v02/artifact_crosscheck.json`
   - `experiments/runs/cleanroom_v02/platform_smoke_user_log_20260719.md`
2. Confirm no long-running training/evaluation jobs are already active.
3. Run `git status --short`. If the worktree is dirty, understand and preserve user changes.
4. Create a checkpoint branch or commit before risky experimentation, after verifying ignored data/model/submission directories remain excluded from Git.
5. Verify the current baseline artifacts and metrics without rebuilding unless necessary.

## Multi-Agent Plan

If multi-agent tools are available, spawn specialist agents with clear scopes and require concise deliverables. Suggested parallel tracks:

1. **Validation Forensics Agents**
   - Inspect OOF predictions, fold dispersion, objective/session grouping, calibration, and error clusters.
   - Identify where current ensemble underperforms baseline, full text, role, or semantic components.
   - Deliver exact files, metrics, and proposed keep gates.

2. **Feature Engineering Agents**
   - Propose leakage-safe features derived only per response/session/objective from training-time artifacts.
   - Focus on transcript trajectory, tutor/student role behavior, objective phrasing, response timing/order, interaction counts, and semantic alignment.
   - Do not propose test-wide aggregation.

3. **Modeling Agents**
   - Explore conservative improvements: calibrated stacking, fold-local meta models, alternative blend weights with nested validation, LightGBM/CatBoost only if install/runtime constraints are acceptable, stronger linear regularization grids, interaction blocks, monotonic/logit transforms.
   - Every proposal must include a compute estimate and rollback plan.

4. **Semantic Agents**
   - Audit BGE usage, pooling, normalization, truncation, prompt formatting, and cached embeddings.
   - Explore offline-only embedding alternatives already vendored or installable without sending data externally.
   - Any new model asset must be license-compatible, packageable, and runtime-feasible.

5. **Sparse/Text Agents**
   - Explore hashing dimensions, n-gram ranges, role-specific vectorizers, objective-text namespaces, char n-grams, TF-IDF alternatives, and SGD/liblinear solver choices.
   - Must use objective-disjoint, validation-session-purged validation.

6. **Packaging/Runtime Agents**
   - Keep package size, runtime, no-network behavior, and official runtime compatibility under control.
   - Audit every proposed dependency and asset before promotion.

7. **Rules/External Research Agents**
   - Use internet search only for official competition/runtime docs, package docs, and public ML ideas.
   - Do not upload or paste competition data anywhere.
   - Summarize usable ideas with citations and explain whether they comply with the competition constraints.

8. **Experiment Scheduler Agent**
   - Build a priority queue of experiments by expected lift, risk, and runtime.
   - Enforce small keep gates before expensive full reruns.
   - Track results in `experiments/runs/top5_<date>/`.

## Experiment Discipline

For every candidate:

1. Save an experiment manifest with code hash, data manifest hashes, parameters, command, start/end time, and outputs.
2. Run a cheap smoke/keep gate first.
3. Promote only if it improves primary log loss and does not regress robustness beyond an explicit, justified tolerance.
4. Compare against:
   - Current ensemble primary log loss `0.5914910112291625`
   - Current robust protocols in `experiments/runs/robust_cleanroom_v02/summary.json`
5. Keep calibration and worst-fold behavior visible. Do not chase a mean-only improvement with bad fold risk.
6. If an idea fails its keep gate, revert or isolate it and move on.

## High-Value Hypotheses To Test

Prioritize these unless code inspection suggests better options:

1. Fold-local calibrated meta-ensemble using component logits and dense diagnostics, with strict nested validation.
2. Objective-family aware calibration or shrinkage that improves worst protocol `semantic_k25_s0`.
3. Char n-gram sparse component for noisy tutor/student text, packaged via hashing to avoid fitted vocabulary bloat.
4. Better semantic interaction features: asymmetry, max/mean pooling over chunks, role-separated semantic alignment, objective-to-student-last-turn similarity.
5. Reliability-weighted blend by fold/protocol diagnostics, but only if learned inside validation without leakage.
6. Session trajectory features that do not aggregate across test rows: early/late role balance, turn counts, content length, question density, hint/explanation markers, student uncertainty markers.
7. Conservative probability calibration: logit clipping, isotonic/platt only if fold-local and robust.
8. Alternative sparse learner settings and pass counts with deterministic seeds.
9. Runtime-safe packaging optimizations if new assets or dependencies are introduced.

## Target Outputs

When a new best candidate exists, produce:

- A new final model under `models/`, with manifest and SHA-256.
- A deterministic ZIP under `submissions/builds/`, with manifest and SHA-256.
- A public download alias in `/opt/sera/app/static/downloads/` named with `submissionN`, for example:
  - `submission2_trace_ace_top5_<short_hash>.zip`
- A public URL:
  - `https://sera.64-227-166-78.sslip.io/static/downloads/<filename>.zip`
- Updated validation tables:
  - Primary log loss, AUROC, Brier, ECE10, fold std, worst fold.
  - Robust protocol table, including worst protocol.
  - Component and ensemble comparison against current candidate.
- Local extracted runtime verification.
- Official pinned runtime verification.
- Platform Smoke result if authenticated access exists.
- A clear verdict:
  - `READY FOR NORMAL` only if local gates and platform Smoke pass for the exact ZIP hash.
  - `NOT READY` otherwise.

Final response must include the public ZIP link, local ZIP path, size, SHA-256, validation deltas, runtime results, platform Smoke status, and the explicit Normal-submission authorization question.

## Stop Condition

Stop when either:

- A clearly stronger, Smoke-passed candidate is ready for user authorization as a Normal submission, or
- The top-5 push hits a defensible plateau and the best current candidate plus all failed high-value hypotheses are documented.

Do not consume Normal quota without user approval.
