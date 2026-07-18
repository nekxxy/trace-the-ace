# Trace the Ace Submission-Readiness Prompt

Work autonomously in `/opt/trace-the-ace` and make the clean-room Trace the Ace model fully submission-ready. Do not merely inspect or report—complete the implementation, training, validation, packaging, and runtime checks. Continue through recoverable failures and make reasonable technical decisions without asking me non-blocking questions.

## Known starting state

- The previous session ended mid cache-integrity audit.
- 30 tests currently pass.
- `data/interim/cache` and the 35,072-row response-view cache exist.
- Sparse/semantic stores, CV results, final model, and submission ZIP do not yet exist.
- No training process is running.
- The directory is not currently a Git repository.
- `docs/history` contains imported prior-project history; it is evidence only and is not the current implementation.
- The current clean-room defaults must earn promotion through fresh validation.

## Safety and operating rules

1. Read `README.md`, `docs/implementation_decisions.md`, relevant scripts/tests, manifests, competition rules, and the official runtime contract before acting.
2. Never expose, print, commit, transmit, or package competition transcripts or test-derived content.
3. Do not use external APIs for transcript processing.
4. Preserve `data/source` unchanged.
5. Reuse manifest-valid expensive artifacts; rebuild anything stale, incomplete, inconsistent, or unverifiable.
6. Send concise progress updates at least every 60 seconds while long work runs.
7. Monitor long-running commands until completion; do not launch them and disappear.
8. Do not weaken validation gates, cherry-pick folds, or describe an unvalidated package as ready.
9. Do not upload a Normal submission without my explicit final confirmation.
10. A platform Smoke submission may be made without asking if authenticated access is already available and it does not consume Normal quota. Never request or expose account credentials.

## A. Recovery and integrity

- Confirm no stale processes are running.
- Finish the interrupted strict-reader/cache audit.
- Verify row counts, schemas, identities, hashes, source-to-view parity, finite dense features, and deterministic rebuild behavior.
- Repair code or rebuild caches if any check fails.
- Run the complete test suite after every material fix.
- Create a local Git recovery checkpoint only after confirming `.gitignore` excludes raw data, caches, models, submissions, secrets, and large assets. Do not push anywhere.

## B. Model stores and validation

- Build the sparse and semantic stores using the current response-view manifest.
- Run leakage-safe objective-disjoint, validation-session-purged sparse and semantic CV.
- Run the repeated semantic-family robustness evaluation required by the project policy.
- Evaluate individual components and the fixed ensemble with log loss as the primary metric; report AUROC, Brier score, ECE, fold dispersion, and worst protocol as diagnostics.
- Check prediction finiteness, bounds, row identity, reproducibility, and leakage constraints.
- If implementation defects or unjustified modeling choices are found, fix them and rerun the affected gates.
- Do not spend a platform submission on a candidate that lacks defensible fresh OOF evidence.

## C. Final training and packaging

- Train the final artifact only after validation passes.
- Expected artifact: `/opt/trace-the-ace/models/final_ensemble_cleanroom_v02.joblib`.
- Build the deterministic ZIP: `/opt/trace-the-ace/submissions/builds/trace_ace_cleanroom_v02.zip`.
- Produce its JSON manifest and SHA-256.
- Audit every ZIP member. Confirm root `main.py`, model artifact, inference modules, licences, notices, and pinned offline BGE assets are present.
- Confirm no data, caches, credentials, absolute paths, network calls, retraining logic, test-wide aggregation, or prohibited logging is included.
- Build twice and verify byte-identical ZIP hashes.

## D. Runtime verification

- Use the pinned official runtime checkout at `/opt/trace-the-ace/vendor/tutoring-outcomes-runtime`, commit `ea9a81755e101b8036e386430c3a2f3d7c655f2e`.
- Run the official local submission test in its container/runtime.
- Run the package on the local smoke fixture.
- Verify exact `response_id` order, required CSV columns, probabilities strictly finite and within `[0,1]`, correct row count, deterministic repeated output, batch invariance, offline model loading, no warnings, exit code `0`, and runtime comfortably within limits.
- Test from the ZIP's extracted contents, not from the development source tree.
- If authenticated platform access exists, upload this exact hash as a Smoke test, monitor it to completion, inspect logs, and verify successful output. Do not silently substitute a rebuilt ZIP afterward.
- If platform access is unavailable, do every local gate and provide the single exact remaining manual Smoke-upload action.

## E. Final handoff

Only call the package **submission-ready** when all required local gates and, where accessible, the platform Smoke gate pass. At the end provide:

- A clear **READY** or **NOT READY** verdict.
- Exact ZIP path, size, entry count, and SHA-256.
- Final model hash.
- CV and robustness score table.
- Test results.
- Official-runtime and smoke results with runtimes.
- Platform job ID if available.
- Live Normal-submission quota if verifiable.
- Remaining risks.
- Exact evidence and artifact paths.
- A statement that the ZIP hash tested is the same hash proposed for submission.

Then stop and ask me one explicit question:

> Do you authorize uploading SHA-256 `<hash>` as the Normal submission?

Do not make the Normal submission until I answer yes.
