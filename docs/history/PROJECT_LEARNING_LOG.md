# Trace the Ace - Project Learning Log

This is the append-only record of what the project team has learned, verified, changed, rejected, and still needs to resolve. It is deliberately separate from the experiment ledger: this file records durable understanding and decisions, while individual model runs will be tracked in a structured experiment table.

## How to maintain this log

- Add one dated section for every day on which meaningful work is done.
- Use local project time (Asia/Kolkata) and ISO dates (`YYYY-MM-DD`).
- Mark statements as **Verified**, **Inference**, **Decision**, **Correction**, **Risk**, or **Open**.
- Never silently rewrite a prior conclusion. Add a correction in the new day's entry and link back to the old conclusion.
- Record unsuccessful approaches and why they failed; negative results are part of the research record.
- Do not paste raw competition transcripts, names, or other row-level competition data here.
- Record source URLs, file names, model versions, licenses, seeds, split versions, and artifact hashes whenever they affect reproducibility.

---

## 2026-07-16 (Asia/Kolkata) - Initial competition, rules, data, and feasibility audit

### Work completed

- Read `Trace_the_Ace_full_overview.md`, `competition_rules.md`, `code_submission_format.md`, and `information_websites_to_see.md`.
- Verified the current competition home, about, rules, code-submission, community-code, and official runtime pages.
- Read and visually checked all four pages of `example_documentation_guide.pdf`.
- Audited the training feature and label tables, both supplied submission-format files, all 22,821 training transcripts, and the locally downloaded external-dataset folders.
- Measured initial group-safe, out-of-fold baselines.
- Audited the current local machine for training and runtime-test feasibility.

### What the competition is actually asking us to do

- **Verified:** For each `response_id`, estimate the probability that the student answered the next assessment question correctly after a tutoring session.
- **Verified:** The inputs for one prediction are a tutoring-session transcript and one learning objective. A session can produce several response rows when several objectives were assessed.
- **Verified:** The target is binary. The required output is a probability, not a hard class.
- **Verified:** The primary metric is binary log loss; lower is better. Leaderboard ROC AUC is secondary and does not determine rank.
- **Verified:** This is a code-execution competition. We submit a ZIP containing inference code and model assets, not a precomputed test-prediction CSV.
- **Verified:** Final prizes are not based only on leaderboard rank. The top 15 teams may submit a four-page write-up, and the judges emphasize educational relevance and generalizability.
- **Verified:** The model-submission deadline is 2026-08-27 23:59 UTC. The top-15 write-up deadline is 2026-09-15.

### Verified training-data facts

| Item | Finding |
|---|---:|
| Training response rows | 35,072 |
| Unique training sessions / transcript files | 22,821 |
| Unique learning-objective IDs and texts | 398 / 398 |
| Positive labels | 24,637 (70.2469%) |
| Negative labels | 10,435 (29.7531%) |
| Total transcript utterances | 6,139,854 |
| Approximate transcript CSV size | 600.9 MB |
| Total transcript content characters | 416,437,335 |
| Median utterances per session | 267 |
| Session utterance range | 15 to 622 |
| Median content characters per session | 18,387 |
| Session content-character range | 337 to 44,556 |
| Tutor utterances | 3,196,001 |
| Student utterances | 2,697,152 |
| Background utterances | 246,701 |
| Feature rows without a transcript | 0 |
| Orphan transcript files | 0 |
| Null cells in core feature/label tables | 0 |

- **Verified:** `response_id` is unique in both feature and label files and the two files join one-to-one with no missing rows.
- **Verified:** `session_id` repeats. Responses per session range from 1 to 10; 14,457 sessions have one response, while 8,364 sessions have multiple responses.
- **Verified:** 3,207 sessions contain a mix of correct and incorrect outcome labels across learning objectives.
- **Verified:** Every `learning_objective_id` maps to exactly one objective text and vice versa in the training data.
- **Verified:** 332 of 398 learning objectives occur in more than one session. Objective difficulty is a strong signal.
- **Verified:** Utterance IDs start at 0 and are contiguous within every audited transcript.
- **Verified:** Transcript timestamps are relative `HH:MM:SS` values, not full datetimes.
- **Verified:** The role distribution is dominated by tutor/student speech but includes a real third role, `background`, in 22,665 sessions.
- **Inference:** The `background` role likely includes slide/system/audio-context material or diarization spillover. It may carry lesson context and must not be discarded without an ablation.
- **Risk:** The samples inspected show ASR noise (`[unclear]`) and occasional apparent speaker-attribution errors. Speaker tags are useful but not perfectly reliable.

### Corrections to the supplied written overview

These differences are not necessarily competition errors; they are schema facts that our code must handle explicitly.

1. **Correction:** The overview documents three feature columns, but the downloaded feature file has four: `response_id`, `session_id`, `learning_objective_id`, and `learning_objective`.
2. **Correction:** The overview calls the label column `correct`; the downloaded label file calls it `is_correct`.
3. **Correction:** The overview says transcript role is either `tutor` or `student`; the downloaded transcripts also contain `background`.
4. **Correction:** The overview describes `utterance_id` as a string; it is stored as an integer in the audited CSVs.
5. **Correction:** The overview describes `timestamp` as a datetime; it is a relative time-of-session field in the audited CSVs.
6. **Decision:** All readers will validate aliases and actual schemas rather than hard-coding only the overview's names.

### Group-safe baseline results

Method: five-fold `StratifiedGroupKFold`, grouped by `session_id`, seed `20260716`. These are local development estimates, not leaderboard scores.

| Model | OOF log loss | OOF ROC AUC | Interpretation |
|---|---:|---:|---|
| Global positive-rate prior | 0.608758 | 0.5000 | Minimum sanity baseline |
| Legal per-sample length/role metadata logistic model | 0.606996 | 0.5377 | Session scale alone is weak |
| Smoothed training-only objective-ID prior | 0.551985 | 0.7067 | Objective difficulty is very strong |

- **Verified:** Learning-objective difficulty is the strongest simple signal found so far.
- **Decision:** The objective-only baseline is necessary for performance and as a control, but it is not the research contribution. Every transcript model must be compared against it and demonstrate added value from conversation content.
- **Risk:** An objective-ID lookup will fail on unseen objective IDs. It needs a text-based objective representation and a global-prior fallback.
- **Verified diagnostic:** Training rows from sessions with more assessed objectives have a higher positive rate.
- **Correction / Rule consequence:** The number of other response rows sharing a test session cannot be used as an inference feature because it depends on other test cases. It is excluded even though it is predictive in training.

### Validation and leakage conclusions

- **Decision:** Never use a random row split. The same transcript would otherwise appear in train and validation through different `response_id` rows.
- **Decision:** The primary split groups by `session_id` and is frozen for model comparison.
- **Decision:** A secondary cold-objective evaluation will test performance when objective IDs or objective groups are unseen.
- **Decision:** Calibration must be trained on genuinely out-of-fold predictions, never on the labels used to fit the underlying model.
- **Decision:** No feature may depend on the composition, aggregates, pseudo-labels, embeddings, or statistics of other test samples.
- **Decision:** Any transcript chunk selection must use only the current row's objective and transcript plus parameters fitted on training data.

### Runtime and submission facts

- **Verified:** Python 3.12 only; model assets must be packaged because inference has no internet access.
- **Verified:** ZIP root must contain `main.py`; it must read from read-only `data/` and write root-level `submission.csv`.
- **Verified:** Output columns must be exactly `response_id,probability`, with one row per required response.
- **Verified:** The supplied 100-row submission format is the smoke-test shape; the 10,508-row file is the full test shape.
- **Verified:** Full runtime limit: 6 hours. Smoke-test limit: 10 minutes.
- **Verified:** Runtime hardware: 24 vCPUs, 220 GB RAM, one NVIDIA A100 with 80 GB VRAM.
- **Verified:** Maximum ZIP size: 60 GB. The process has no root filesystem access.
- **Verified:** Test transcript text, objective text, token counts, dataset summaries, or other test-data information must not be printed. Logging is limited to 500 lines and 500 characters per line.
- **Verified:** Current runtime packages include PyTorch, Transformers, sentence-transformers, LightGBM, scikit-learn, PEFT, vLLM, pandas, and Polars. Exact locked versions must be tested in the official container before submission.
- **Verified:** Full submissions are limited to three per seven days. Smoke tests, cancelled jobs, and failed jobs do not consume that quota.

### External-data and model licensing audit

The rule is stricter than “publicly downloadable”: resources must permit commercial use and support an openly releasable solution. We will maintain source and license evidence for every external resource.

| Local resource | Verified/known license state | Current decision |
|---|---|---|
| `Grade School Math (GSM8k)` | MIT | Eligible candidate, but only indirectly relevant |
| `FairytaleQA` | Apache-2.0 | Eligible, low relevance to math tutoring outcomes |
| `Bridge` | CC BY-NC 4.0 | Exclude from prize-eligible training |
| `DrawEduMath` | Platform catalog lists CC BY-NC-SA 4.0 | Exclude from prize-eligible training |
| `SciQ` | CC BY-NC 3.0 | Exclude from prize-eligible training |
| `SemEval` folder | Contents match the TalkMoves classroom corpus; upstream is CC BY-NC-SA 4.0 | Exclude from prize-eligible training |
| `TalkMoves` folder | Contents match SemEval-2013 Task 7 (BEETLE/SciEntsBank), not TalkMoves classroom transcripts; platform catalog lists SemEval as CC BY-SA 3.0 | Quarantine pending organizer confirmation on share-alike compatibility |
| `MRBench` | Repository declares CC BY-SA 4.0 but it builds on Bridge, which is non-commercial | Exclude unless organizers explicitly clear it |
| `EssayJudge` | Platform catalog lists Apache-2.0 | Eligible candidate, but low direct relevance |

- **Correction:** The local `SemEval` and `TalkMoves` folder names appear semantically swapped relative to their contents. They have not been renamed because raw source folders should remain immutable.
- **Decision:** Do not download more external datasets now. The immediate performance bottleneck is modeling and validating the competition data, not data volume.
- **Decision:** Initial pretrained-model candidates are Apache-2.0 resources: ModernBERT-base (8,192-token encoder), Qwen3-Embedding-0.6B (32K embedding context), all-MiniLM-L6-v2 (fast short-chunk baseline), and Qwen2.5-7B-Instruct (optional structured annotation/direct scoring). Each exact revision will be registered before use.

### Later clarification from the platform dataset catalog — 2026-07-16

The user supplied screenshots of the K-12 AI Infrastructure dataset catalog, and the public catalog was checked again. This corrected part of the first local-file-only license audit.

- **New information:** EssayJudge is listed as Apache-2.0, so it is an eligible external-data candidate. It remains low priority because its lexical-quality target is not the competition target.
- **New information:** The official catalog lists SemEval as CC BY-SA 3.0 and TalkMoves as CC BY-NC-SA 4.0. This reinforces that the local `TalkMoves` folder contains SemEval material and the local `SemEval` folder contains TalkMoves material.
- **What was previously wrong:** EssayJudge was initially quarantined because the local files did not expose a clear license. The platform listing resolves that uncertainty in favor of Apache-2.0.
- **What remains unresolved:** CC BY-SA permits commercial use, but the competition asks for a permissive open license and an MIT-releasable winning solution. SemEval therefore stays quarantined until the organizers confirm that its share-alike obligation is compatible with this competition.
- **What remains unsafe:** MRBench is listed as CC BY-SA 4.0, but its documented use of Bridge creates an upstream CC BY-NC concern. A catalog listing alone does not remove that conflict.
- **Decision:** Being listed on the platform is not a blanket exception to the Trace the Ace external-data rule. The prize-safe allowlist is currently GSM8K (MIT), FairytaleQA (Apache-2.0), and EssayJudge (Apache-2.0). We will not train on any NC resource, and will not train on BY-SA resources without written organizer clearance.
- **Decision:** No download is needed now because the audited local folders already contain these catalog datasets. If a clean re-download is later needed, use the platform-hosted version, record its version/date/URL and SHA-256 hash, and keep it immutable.

### Research direction learned from the competition references

- **Verified from prior work:** Dialogue text can predict student outcomes, and explicit tutor-move information can add signal.
- **Verified from prior work:** Outcome prediction is more tractable than future tutor-move prediction.
- **Verified from prior work:** Instructionally supportive behaviors, probing, feedback, and explanation can correlate with outcomes, but effects vary by dataset.
- **Decision:** We will model student evidence and tutor actions separately, then test their incremental predictive value with ablations.
- **Risk:** This observational dataset supports predictive association, not causal claims about a tutoring strategy causing learning. The write-up must use causal language only if the analysis design justifies it.

### Local development environment

| Resource | Finding |
|---|---|
| CPU | AMD Ryzen 5 4600H, 6 cores / 12 threads |
| RAM | 11.4 GiB total |
| Local NVIDIA GPU | None detected |
| Free C: drive space | about 208 GiB |
| Docker | Not installed/detected |
| `just` command runner | Not installed/detected |
| `uv` and Git | Available |

- **Decision:** Local hardware is adequate for audits, Parquet conversion, TF-IDF, logistic regression, and small CPU experiments.
- **Risk:** Long-context embedding and transformer fine-tuning require cloud GPU access or another GPU machine.
- **Open:** Docker Desktop and `just` must be installed before official local container tests.

### Decisions carried into implementation

1. Freeze a session-grouped validation split before serious model tuning.
2. Build a schema-validated, cached transcript table instead of repeatedly opening 22,821 CSVs.
3. Preserve all three speaker roles and test removal/merging only through ablation.
4. Establish objective-only and legal metadata baselines before transcript models.
5. Add word/character TF-IDF transcript models before spending GPU time.
6. Use objective-conditioned long-context and key-moment models, with the full transcript as a comparison.
7. Optimize and report log loss first; treat AUC as diagnostic.
8. Track calibration, runtime, licenses, seeds, and negative results from the beginning.
9. Quarantine non-commercial and unclear external datasets.
10. Build the submission runner early enough that runtime constraints influence model selection.

### Open questions for the next work session

- What is the strongest legal word/character TF-IDF baseline from transcript text, objective text, and their interaction?
- How much does the final quarter of a session contribute versus the full transcript?
- Can objective-conditioned chunk retrieval beat naive truncation?
- Does `background` content improve log loss after controlling for objective and transcript length?
- How many test objectives are unseen cannot be known from local files; the pipeline needs a robust fallback by design.
- Which cloud GPU environment will be used for embedding extraction and fine-tuning?
- Should the organizers be asked to clarify whether CC BY-SA models/data derived from non-commercial sources can ever be prize eligible? Until clarified, they remain excluded.

### Sources checked on this date

- Local: `Trace_the_Ace_full_overview.md`
- Local: `competition_rules.md`
- Local: `code_submission_format.md`
- Local: `example_documentation_guide.pdf`
- Competition: https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/
- K-12 AI Infrastructure dataset catalog: https://platform.k12-ai-infrastructure.org/
- Official runtime: https://github.com/drivendataorg/tutoring-outcomes-runtime
- Ikram, Scarlatos, and Lan (2025): https://arxiv.org/abs/2507.06910
- Scarlatos, Baker, and Lan (2025): https://arxiv.org/abs/2409.16490
- ModernBERT model card: https://huggingface.co/answerdotai/ModernBERT-base
- Qwen3 Embedding model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen2.5 model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- GSM8K model card/repository: https://huggingface.co/datasets/openai/gsm8k and https://github.com/openai/grade-school-math
- FairytaleQA repository: https://github.com/uci-soe/FairytaleQAData
- Bridge dataset card: https://huggingface.co/datasets/rose-e-wang/bridge
- TalkMoves repository: https://github.com/SumnerLab/TalkMoves
- SciQ dataset card: https://huggingface.co/datasets/allenai/sciq

## 2026-07-16 (Asia/Kolkata) - Foundation, grouped baselines, and first deployable transcript model

### Work completed after plan approval

- Initialized the Git repository and added a conservative `.gitignore` that prevents competition transcripts, feature/label tables, external datasets, caches, model assets, submissions, and temporary files from being committed.
- Added the installable `trace_ace` Python package, `pyproject.toml`, and a resolved `uv.lock` for reproducible local dependencies.
- Added the project configuration and external-resource license registry under `configs/`.
- Built a raw competition-data manifest with path, category, byte size, modification time, and SHA-256 for every supplied competition file.
- Built deterministic Parquet caches for utterances, session features, session text views, response-level modeling data, and frozen fold assignments.
- Added an evaluation harness for log loss, ROC AUC, Brier score, ten-bin expected calibration error, pooled OOF metrics, and fold-level metrics.
- Trained group-safe global, metadata, objective-ID, objective-text, full-session word TF-IDF, and closing-quarter character TF-IDF baselines.
- Added an appendable experiment ledger plus per-run reports and OOF predictions.
- Fitted the promoted full-data sparse word model and created two offline packages: an objective-prior fallback and the transcript-based word TF-IDF candidate.
- Added an inference batch-invariance test, local 100-row smoke tests, ZIP-structure checks, and a formal milestone verifier.

### Reproducibility and cache results

| Item | Result |
|---|---:|
| Files in raw competition manifest | 22,825 |
| Bytes in raw competition manifest | 603,773,729 |
| Raw manifest digest | `38a63c7e794277f252327dc36dcd7bbbe312a7f404ce9912ccdc72c9333d8e38` |
| Cached sessions | 22,821 |
| Cached utterances | 6,139,854 |
| Exact transcript content characters | 416,437,335 |
| First manifest/cache/fold build | about 129 seconds |
| Idempotent rerun using current cache | about 9.4 seconds |

- **Correction:** The earlier 423-million-character value was a rough estimate. The validated streaming cache reports exactly 416,437,335 content characters.
- **Verified:** Every session belongs to exactly one of the five folds. No transcript/session crosses a training-validation boundary.
- **Verified:** The frozen folds cover all 35,072 responses and use seed `20260716`.
- **Verified:** The second foundation run detected an unchanged source digest and skipped rebuilding the expensive transcript cache.

### Reproducible grouped OOF results

All results use five-fold `StratifiedGroupKFold`, grouped by `session_id`. Every learned vectorizer and model is fitted only on the training portion of each fold.

| Model | OOF log loss | OOF AUC | Brier | ECE-10 |
|---|---:|---:|---:|---:|
| Fold-trained global prior | 0.608798 | 0.4932 | 0.209023 | 0.00002 |
| Legal metadata logistic | 0.599717 | 0.5834 | 0.205348 | 0.00408 |
| Smoothed objective-ID prior | 0.551985 | 0.7067 | 0.185812 | 0.00915 |
| Objective-text TF-IDF | 0.551396 | 0.7070 | 0.185543 | 0.00335 |
| Full-session word TF-IDF + objective text | **0.532189** | **0.7384** | **0.177879** | **0.00413** |
| Closing-quarter character TF-IDF + objective text | 0.548694 | 0.7136 | 0.184440 | 0.01242 |
| Fixed 50/50 word-character blend | 0.535148 | 0.7338 | 0.179052 | 0.00475 |

### What we learned from the first transcript models

- **Verified:** Full transcript language adds substantial signal beyond objective difficulty. The word model improves log loss by `0.019796` over the objective-ID baseline, about a 3.59% relative reduction.
- **Verified:** The full-session word model beats the objective-text model in every fold, so the gain is not driven by one favorable partition.
- **Verified:** The final quarter alone contains useful information, but the character model's `0.548694` is much weaker than the full-session word model.
- **Negative result:** A fixed equal blend of word and character predictions is worse than the word model alone. The character model is not promoted in its current form.
- **Correction:** The first metadata audit used a smaller feature set and reported about `0.6070`. The implemented legal per-sample metadata model includes role proportions, duration, objective length, `[unclear]` rate, and other transcript-shape features and reaches `0.599717`.
- **Correction:** A pooled constant prior gives AUC 0.5, but the stricter fold-trained global priors vary slightly across folds and produce pooled AUC `0.4932`. This is expected and does not make the model meaningfully discriminative.
- **Decision:** Promote the full-session word TF-IDF model as the first transcript-based submission candidate. Do not promote the equal word-character blend.
- **Decision:** External datasets remain unused. The competition-only transcript gain is already strong, so external data must still demonstrate incremental grouped-CV value before entering training.

### Runtime and engineering learnings

- The initial five-fold word-plus-character run took about 53 minutes on the local Ryzen CPU and peaked below 4 GB RAM.
- The launching shell reached its 30-minute command limit, but the Python worker continued and completed successfully. The original implementation wrote results only at the end, which made the run unnecessarily fragile.
- **What was previously wrong:** The text baseline was initially all-or-nothing. It now writes atomic per-model/per-fold checkpoints and resumes completed folds.
- The promoted full-data word model fitted in about 194 seconds.
- The final sparse training matrix has shape `35,072 × 61,297`.
- The trained artifact is 1,195,372 bytes with SHA-256 `b337062bb9bf622726b5c7057ca2bd52859c1028edfa76c7e3fcd2c8f2128126`.
- The transcript submission ZIP is 1,202,970 bytes, far below the 60 GB limit.
- Both packages passed the supplied 100-response smoke-format test locally without emitting stdout or stderr.
- Eight unit tests pass. The formal milestone verifier passes 23 checks covering manifests, cache row counts, fold isolation, targets, licenses, metrics, artifact hash, ZIP contents, and size limits.

### Current submission status

| Package | Purpose | Local status |
|---|---|---|
| `submission_builds/objective_prior_baseline.zip` | Minimal fallback and runtime debugging | 100-row local smoke passed |
| `submission_builds/word_full_tfidf_baseline.zip` | First promoted transcript model | 100-row local smoke passed |

The promoted package has not been uploaded. Before spending a full-submission quota, test it in the official runtime container and then in the platform smoke environment. Docker is not currently installed on the local machine, so the official-container check remains open.

### Next modeling questions

- Can role-separated tutor/student word features beat the combined transcript?
- Does full word text plus a smoothed objective prior outperform either component without degrading calibration?
- Are opening, middle, and closing transcript windows complementary to the full-session model?
- Can a stateless hashed feature cache shorten fold-safe sparse experiments?
- Do frozen long-context embeddings improve on `0.532189` enough to justify GPU work?

## 2026-07-16 (Asia/Kolkata) - First platform smoke test and runtime-aligned rebuild

### Platform result

- Uploaded `word_full_tfidf_baseline.zip` as smoke job `1677`.
- **Completed successfully** with smoke score `0.4022` and exit code `0`.
- The submitted `main.py` completed inference in about 3.6 seconds inside the official CUDA 12.9 container.
- The runtime found the correct root-level `main.py`, all three assets, and the generated `submission.csv`.
- **Important interpretation:** The smoke environment uses 100 responses drawn from training data, and the full sparse model was trained on those responses. The `0.4022` score is therefore optimistic and is only a functional check, not a valid estimate of private-test or leaderboard performance.

### Warning discovered

The official runtime emitted `InconsistentVersionWarning` for `TfidfTransformer`, `TfidfVectorizer`, and `LogisticRegression`:

- original artifact build: scikit-learn `1.6.1`;
- official runtime: scikit-learn `1.8.0`.

The artifact loaded and scored, but scikit-learn explicitly warns that cross-version unpickling can produce invalid results. A successful smoke job does not remove that risk.

### Correction implemented

- Added `.python-version` selecting Python 3.12.
- Pinned project scikit-learn to `1.8.0` and refreshed `uv.lock`.
- Created a clean `.venv` with Python `3.12.8` and scikit-learn `1.8.0`.
- Added Python, scikit-learn, NumPy, and joblib build versions to the trained artifact metadata.
- Added a hard inference-time scikit-learn version check so a mismatched artifact fails explicitly.
- Retrained the promoted model without changing its data, features, or model structure.
- Rebuilt both ZIP packages and reran the 100-row local smoke test under the aligned environment with no warnings or output.
- All eight unit tests pass and the artifact verifier now also enforces the required Python/scikit-learn build versions.

### Replacement artifact

| Item | Value |
|---|---|
| Python | 3.12.8 |
| scikit-learn | 1.8.0 |
| NumPy | 2.5.1 |
| joblib | 1.5.3 |
| Model SHA-256 | `3dd0481ac303c097a57275a443ce03a5029a9531fa3d28441e83ce6219ff0a84` |
| Replacement upload file | `submission_builds/word_full_tfidf_runtime180.zip` |
| Replacement ZIP SHA-256 | `8f1da10ff54f6463ba9c5b2b51bd000d8659d52760fbaead282c03046c4feee2` |

- **Decision:** Do not use the original job-1677 archive for a Normal submission.
- **Next action:** Upload the replacement ZIP for one more platform smoke test. If it completes without `InconsistentVersionWarning`, use that exact ZIP for the first Normal submission.

### Replacement smoke confirmation

- Uploaded the runtime-aligned replacement as smoke job `1682`.
- **Completed successfully** with exit code `0` and smoke score `0.4022`.
- Inference completed in about 3.3 seconds in the official container.
- No `InconsistentVersionWarning` or other Python/model warning appeared.
- The identical `0.4022` score confirms that aligning the serialization environment did not change the model's predictions on the smoke set.
- **Clarification:** The competition metric is log loss, so a lower number is better. However, the smoke score is calculated on 100 examples drawn from training data and is not a valid leaderboard or held-out-performance estimate.
- **Decision:** Runtime validation is complete. Submit the exact runtime-aligned ZIP as the first Normal submission to obtain the first genuine leaderboard measurement.

## 2026-07-16 (Asia/Kolkata) - Leaderboard failure diagnosis and v0.2 robust ensemble

### Leaderboard evidence that changed the modeling strategy

- **Verified:** Normal submission job `1685` completed successfully and scored public log loss `0.6181`, public AUROC `0.6129`, and rank `75` at the time observed.
- **Verified:** The leading public entry shown at the same time had log loss `0.6013` and AUROC `0.6309`.
- **Correction:** The original session-grouped OOF estimate (`0.532189` log loss) was not representative of the leaderboard. It prevented transcript leakage, but most validation rows still used learning objectives that the model had seen in other sessions. Objective difficulty and vocabulary therefore made the split much easier than the public test regime.
- **Decision:** Do not tune further against the original session-grouped score. It remains useful for measuring seen-objective behavior, but model promotion now requires a second, harder protocol designed around objective shift.

### Objective-disjoint validation protocol

- **Implemented:** Five-fold `StratifiedGroupKFold` grouped by `learning_objective_id`, seed `20260716`.
- **Implemented:** After assigning objective-disjoint validation folds, remove from each training fold every session that appears in that fold's validation rows. This prevents both objective overlap and transcript overlap.
- **Verified:** A simple stateless hashed transcript/objective model reached hard-validation AUROC `0.613226`, extremely close to the observed public AUROC `0.6129`.
- **Inference:** This close match is strong evidence that unseen or shifted objectives explain much of the public failure. It is not proof of the hidden split construction, so the protocol is described as a leaderboard-matched stress test rather than the official split.
- **Verified diagnostic:** On the original OOF predictions, the full word model's log loss was about `0.6427` for objectives unseen in the training fold and about `0.5245` for objectives with at least 500 training examples. The prior validation average was dominated by the easier frequent-objective regime.
- **Verified:** Probability shrinkage toward each fold's training prior materially improved hard-validation log loss while leaving ranking unchanged. The original models were overconfident under objective shift.

### New competition-only feature work

- Built `session_role_texts.parquet` for all 22,821 sessions with tutor text, student text, opening text, closing tutor text, and closing student text.
- Built `session_behavior.parquet` with 16 deterministic per-session counts, means, and pedagogy proxies, followed by eight legal per-sample ratios.
- Built five reusable role-specific hash matrices plus response-order guards.
- Built `response_objective_context.parquet` for all 35,072 responses. The extractor independently matches one row's objective against only its own transcript, retains neighboring turns, and records ten coverage/position statistics.
- **Verified:** Objective matching found at least one relevant line for `99.23%` of training responses. The median selected context was 45 lines from a median 266-line session.
- **Verified:** 8,364 sessions contain multiple assessed objectives, and 3,207 sessions contain mixed correct/incorrect targets. This explains why objective-conditioned views are conceptually necessary even when their standalone model is not strongest.

### Hard-validation ablations and negative results

All values below use the objective-disjoint split with validation-session purge. `Shrunk loss` is the best OOF-only linear probability blend with the fold training prior; it is reported for diagnosis, not automatically transferred to the final ensemble.

| Model/view | Raw log loss | AUROC | Shrunk loss | Conclusion |
|---|---:|---:|---:|---|
| Hash transcript + objective, alpha `3e-5` | 0.600716 | 0.613226 | 0.591099 | First leaderboard-matched control |
| Tutor + student + objective | 0.596654 | 0.614181 | 0.590579 | Role separation helps |
| Role/objective + behavior/alignment | 0.597195 | 0.619776 | 0.589000 | Strong sparse component |
| Role/retrieval/objective + dense, alpha `1e-4` | 0.592476 | 0.620414 | 0.588714 | Small standalone gain |
| BGE semantic interaction + dense, `C=0.1` | 0.588525 | 0.624163 | 0.588467 | Strong complementary component |
| Final fixed 25/25/50 ensemble | **0.584582** | **0.634437** | not needed | Promoted v0.2 candidate |

- **Negative result:** Standalone objective-retrieved word text reached only AUROC `0.5873`; it did not beat full transcript or role models.
- **Negative result:** Retrieved character n-grams reached only AUROC `0.5675`; the ASR-robust character hypothesis did not pay off in this form.
- **Negative result:** Closing-student text plus objective reached AUROC `0.5783`; the final quarter alone discards too much context.
- **Negative result:** BGE context/objective embeddings without behavior features peaked near AUROC `0.5775`.
- **Control:** The deterministic behavior/retrieval dense features without BGE peaked near AUROC `0.6089`. The combined semantic/dense model's `0.6242` therefore reflects real semantic complementarity, not just the dense controls.
- **Learning:** The full-transcript model has weaker standalone hard-validation AUROC (`0.5992`) but improves the ensemble because its errors differ from the role and semantic models.
- **Risk:** A weight optimizer fitted to all OOF rows produced AUROC `0.6342`, while leave-one-objective-fold-out weight selection produced `0.6260`. The promoted weights are deliberately rounded (`0.25`, `0.25`, `0.50`) and the `0.6344` estimate must still be treated as optimistic.
- **Decision:** Exclude the standalone retrieval sparse component from v0.2 because the nonnegative ensemble optimizer assigned it zero weight. Keep objective retrieval only inside the semantic feature path.

### Provider and external-resource conclusions

- **Verified from the competition description:** The challenge combines Eedi short typed chats and Third Space Learning long voice/ASR sessions.
- **Verified locally:** The 22,821 supplied training transcripts behave like the long voice/ASR provider: median duration is roughly 43 minutes, `[unclear]` is frequent, and examples include spoken/audio artifacts.
- **Inference:** Provider shift may be another hidden-test risk. The hidden provider composition is not disclosed, so do not present this as confirmed.
- **License decision:** `Eedi/Question-Anchored-Tutoring-Dialogues-2k` is marked CC BY-NC 4.0 and its model card directs commercial users to contact Eedi. It was not downloaded or used for training. Written commercial permission and organizer clearance would be required first.
- **External model used:** `BAAI/bge-small-en-v1.5`, MIT license, 384 dimensions, packaged for offline inference. Model weight SHA-256: `ea1d11a3f23d14fe09fc1826fc7944e89c09a634d2217d57a21dd136805ee3e8`.
- **Verified runtime compatibility:** Official runtime repository commit `ea9a81755e101b8036e386430c3a2f3d7c655f2e` includes Python 3.12, scikit-learn 1.8.0, sentence-transformers 5.5.0, Transformers 4.57.6, PyTorch, and an A100 runtime.

### Promoted v0.2 artifact and verification

| Item | Value |
|---|---|
| Model artifact | `models/final_ensemble_v02.joblib` |
| Artifact size | 3,012,930 bytes |
| Artifact SHA-256 | `14f2f7dd5298a2c7ebd14800f5d5f3263c15f90246e8533f75a44937d2e1b77d` |
| Upload ZIP | `submission_builds/final_ensemble_v02.zip` |
| ZIP size | 81,118,668 bytes |
| ZIP SHA-256 | `c6d64f694fe3e983826af904e5cd0a39a91091eadaafdb97a7fe1a42df823827` |
| Fixed blend | 25% full transcript, 25% role/objective/dense, 50% BGE semantic/dense |

- **Verified:** Twelve unit tests pass.
- **Verified:** The formal milestone verifier still passes all 25 checks.
- **Verified:** Standalone inference reproduced cached full-data model probabilities within maximum absolute error `3.1e-8` on a 20-response end-to-end comparison.
- **Verified:** Final inference was batch-invariant within maximum absolute error `1.6e-8` on real training samples.
- **Verified:** The 100-row local smoke fixture completed with valid probabilities and no stdout/stderr. Its in-sample log loss was `0.517774` and AUROC `0.76637`; these are functional diagnostics only, not held-out estimates.
- **Verified:** A clean extract-and-run of the actual ZIP produced byte-identical prediction CSV content to the staged run. The archive includes the otherwise-empty `assets/bge-small-en-v1.5/2_Normalize/` directory required by the model module configuration.
- **Open:** Docker and `just` are not installed locally, so the official Docker image was not run. The platform smoke test is the required authoritative runtime check.
- **Decision:** Do not make another Normal submission before the v0.2 platform smoke test completes cleanly. A top-one outcome is a goal, not a guarantee; the strict local ensemble now reaches the reference top AUROC, but public log loss can still differ because the hidden distribution is unknown.

### Exact next action

1. Upload `submission_builds/final_ensemble_v02.zip` as a **Smoke test**, not a Normal submission.
2. Use private note: `v0.2 objective-shift robust | full hash 25 + role/behavior 25 + BGE semantic 50 | hard CV LL 0.58458 AUC 0.63444`.
3. Confirm root extraction, BGE offline loading, absence of warnings, `submission.csv` creation, exit code `0`, and smoke runtime below 10 minutes.
4. If and only if the smoke test is clean, use the exact same ZIP for one Normal submission.
5. Record the new job ID, runtime, public log loss, public AUROC, rank, and the public-minus-hard-CV gap here before changing any model.

### Additional sources checked

- BGE model card and MIT license: https://huggingface.co/BAAI/bge-small-en-v1.5
- all-MiniLM reference and Apache-2.0 license: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Eedi QATD2k card and commercial-use warning: https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k
- Official runtime source: https://github.com/drivendataorg/tutoring-outcomes-runtime

## 2026-07-16 (Asia/Kolkata) - v0.2 platform smoke job 1708

### Platform result

- Uploaded `submission_builds/final_ensemble_v02.zip` as smoke job `1708`.
- **Verified:** Job status `Completed`, exit code `0`, smoke log loss `0.5178`.
- **Verified:** Inference ran from approximately `17:18:03.698` to `17:18:13.557`, about `9.86` seconds, far below the ten-minute smoke limit.
- **Verified:** The runtime found all 15 ZIP entries, including the empty `2_Normalize/` model-module directory, loaded the packaged BGE model offline, wrote `submission.csv`, and emitted no Python/model warning.
- **Verified parity:** The platform smoke score rounds exactly from the local smoke log loss `0.5177735669`. This confirms that local and official-runtime preprocessing/model predictions agree on the smoke fixture.

### Correct interpretation of the apparently worse score

- **Verified:** `0.5178` is numerically worse than the earlier v0.1 smoke score `0.4022`; lower log loss is better.
- **Correction:** This does not establish that v0.2 will be worse on the hidden leaderboard. The platform documents that smoke data are 100 responses sampled from training data. The v0.1 TF-IDF model was fitted directly on those rows' sessions/objectives and its unusually low `0.4022` was an in-sample memorization score.
- **Learning:** v0.2 is deliberately more regularized and optimized for unseen-objective generalization. It sacrifices performance on already-seen training rows in exchange for materially stronger objective-disjoint OOF performance.
- **Decision:** Use smoke score only as a functional/runtime check. Use objective-disjoint, session-purged OOF evidence to decide whether a scarce Normal submission is justified.
- **Decision:** Job `1708` clears the runtime gate. The exact same ZIP is approved for one Normal submission; its true value must be measured by the public leaderboard, not by the smoke score.

## 2026-07-16 (Asia/Kolkata) - v0.2 public result, rank 16, and top-1 redirection

### Normal submission result

- **Verified:** The exact v0.2 ZIP completed the Normal platform job with exit code `0`.
- **Verified:** Inference ran from approximately `17:34:27.227` to `17:38:08.334`, about `221.1` seconds.
- **Verified:** Public log loss is `0.6081`, public AUROC is `0.6147`, and the entry moved from rank `75` to rank `16` after two Normal submissions.
- **Verified:** The prior public result was log loss `0.6181` / AUROC `0.6129`. v0.2 therefore improved log loss by `0.0100` and AUROC by `0.0018`.
- **Verified from the live leaderboard:** Current rank 15 is `0.6080`, rank 8 is `0.6060` / `0.6145`, and rank 1 is `0.6013` / `0.6309`.
- **Learning:** v0.2's main public gain came from probability robustness/calibration, not a large improvement in ranking. The objective-disjoint hard-CV AUROC `0.6344` did not transfer directly to public AUROC `0.6147`.
- **Correction:** Random objective-ID-disjoint validation remains useful but cannot remain the sole promotion test. It permits semantically related objectives to cross folds and its per-fold difficulty is highly variable.

### Post-result diagnostics

- **Verified:** v0.2 hard-fold log loss ranges from `0.5554` to `0.5969`; hard-fold AUROC ranges from `0.5946` to `0.6986`. The pooled score hides substantial objective-family instability.
- **Verified:** Pooled OOF optimization changes the fixed `25/25/50` weights to approximately `22.2/29.6/48.2` but improves loss by only `0.000054` (`0.584582` to `0.584527`). The rounded blend is already essentially optimal on that validation set.
- **Negative result:** Cross-fitted Platt scaling worsens hard loss to `0.585630`.
- **Negative result:** Cross-fitted beta calibration worsens hard loss to `0.585557`.
- **Negative result:** Learned logistic stacking of component predictions, disagreements, and session metadata is worse than the fixed blend; the best simple form is roughly `0.5877`.
- **Negative result:** Shrinking the ensemble toward the hard-fold training prior does not help; the best weight is 100% v0.2.
- **Negative result:** A BGE k-nearest-objective difficulty prior peaks at standalone hard loss `0.600951` / AUROC `0.579096` and receives zero useful blend weight.
- **Decision:** Do not use a Normal submission for a calibration-only transform, a tiny blend-weight adjustment, or the semantic-neighbor prior.

### New strategy

- **Decision:** Add repeated semantic-family-disjoint and provider/style-proxy stress tests with session purge before training the next submission candidate.
- **Decision:** The highest-priority new model is a multi-view BGE evidence model over role-specific objective chunks, tutor-question/student-answer windows, tutor-feedback windows, and opening/closing evidence.
- **Decision:** Add ordered student-mastery and tutoring-move trajectory features rather than more flat counts.
- **Decision:** Test `Qwen/Qwen3-Embedding-0.6B` only after the multi-view BGE design shows robust value; it is an Apache-2.0, instruction-aware long-text embedding model, but local CPU cost is material.
- **Rule consequence:** Competition transcripts stay local. Do not send them to external APIs or third-party annotation services.
- **Submission policy:** Assume only one Normal submission remains in the current seven-day window. Use it only for a materially new v0.3 that passes the documented promotion gates and a clean smoke test.
- **Documentation:** The full experiment order, thresholds, risks, and submission schedule are in `TOP_1_EXECUTION_PLAN.md`.

## 2026-07-17 (Asia/Kolkata) - V100/V110 robust validation baseline

### What was implemented

- Added `src/trace_ace/robust_validation.py` and `scripts/run_robust_validation.py`.
- Added deterministic semantic-family clustering of the 398 objective descriptions using the frozen BGE objective embeddings.
- Added four semantic-family-disjoint validation protocols using 25, 50, 50, and 80 clusters across different cluster/fold seeds.
- Every fold holds out complete semantic families and objective IDs, then purges every training row whose session appears in validation.
- Added a regime scorecard for fold, transcript length, ASR uncertainty, background rate, objective frequency, objective-retrieval coverage, and the number of response rows associated with a training session.
- Added tests for deterministic assignments, complete folds, semantic-family disjointness, and scorecard construction. The full suite now passes all 14 tests.

### Full robust run

| Item | Value |
|---|---|
| Run ID | `20260716T183434Z_robust_validation` |
| Runtime | approximately 614 seconds including one-time loading of the large sparse caches |
| OOF predictions | `experiments/runs/20260716T183434Z_robust_validation/oof_predictions.parquet` |
| Regime scorecard | `experiments/runs/20260716T183434Z_robust_validation/regime_scorecard.csv` |
| Objective-family assignments | `experiments/runs/20260716T183434Z_robust_validation/objective_family_assignments.parquet` |

| Protocol | v0.2 log loss | v0.2 AUROC |
|---|---:|---:|
| `semantic_k25_s0` | 0.583099 | 0.639570 |
| `semantic_k50_s0` | 0.587578 | 0.624048 |
| `semantic_k50_s1` | 0.582543 | 0.640916 |
| `semantic_k80_s0` | 0.586927 | 0.625124 |
| **Mean** | **0.585037** | **0.632414** |
| **Median** | **0.585013** | **0.632347** |
| **Worst protocol** | **0.587578** | **0.624048** |

- **Verified:** The fixed v0.2 ensemble beats every individual component on log loss in all four semantic-family protocols.
- **Verified:** Protocol variability remains material. Objective clustering seed/resolution changes AUROC by roughly `0.017`, so one semantic split is still not trustworthy by itself.
- **Verified:** The pooled robust optimum is approximately `25.2%` full transcript, `24.1%` role/dense, and `50.7%` semantic/dense. This is effectively identical to the deployed `25/25/50` blend.
- **Verified stability:** Learning weights from three protocols and evaluating the fourth makes held-out loss worse in all four leave-one-protocol-out tests.
- **Decision:** Freeze the v0.2 weights. No further weight-only or calibration-only experiment is eligible for a Normal submission.

### Failure regimes and modeling consequences

- **Actionable failure:** Low objective-retrieval coverage is the worst legal per-sample regime, with mean loss about `0.6264`, worst loss `0.6298`, mean AUROC `0.6063`, and worst AUROC `0.5960`.
- **Actionable failure:** The longest transcript quintile has mean loss about `0.6025` and worst loss `0.6059`. A single compressed 256-token context is inadequate for the longest sessions.
- **Actionable failure:** The rarest objective-frequency quintile has mean loss about `0.6017`; the most frequent held-out-objective quintile is also unstable and has the weakest ranking. Frequency alone does not supply a safe correction.
- **Diagnostic only:** Training sessions associated with one response row have very high loss around `0.649`. The count of other test responses sharing a session is prohibited as an inference feature, so this result may guide research but cannot be used directly.
- **Learning:** The two largest actionable weaknesses, low lexical retrieval coverage and long transcripts, both support multi-view semantic evidence extraction. The next model must retrieve role-specific evidence and multiple moments instead of expanding the current global blend.

### Decision and next action

- V100 (repeated semantic-family validation) and V110 (regime scorecard) are complete.
- Start E200/E210: role-specific, ordered, multi-view BGE evidence caches and hard OOF models.
- v0.3 must beat the frozen v0.2 baseline across these same four protocols before packaging or smoke testing.
