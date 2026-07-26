# Trace the Ace

Reproducible machine-learning project for the [Trace the Ace tutoring-outcomes competition](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/).

The task is to predict the probability that a student answers an aligned follow-up question correctly using a tutoring-session transcript and learning-objective metadata.

## Current status

**See `experiments/improvements/README.md` for the live scoreboard and
`CLAUDE.md` / `docs/agent_handoff.md` for full context — this section is a
snapshot, not the source of truth.**

- All official training tables and 22,821 transcript files are present and validated.
- Raw source downloads are preserved separately from canonical working inputs.
- The official runtime is pinned locally at commit `ea9a81755e101b8036e386430c3a2f3d7c655f2e`.
- Python serialization dependencies now match the official runtime, including scikit-learn 1.8.0 and NumPy 2.2.6.
- Memory-bounded caches, leakage-safe folds, sparse models, offline BGE semantic features, ensemble inference, tests, and deterministic packaging are implemented.
- Primary objective-disjoint CV, all four semantic-family robustness protocols,
  and a session-disjoint hardening protocol are computed for every candidate.
- **Best real submission: v06** (bge-base semantic encoder) — real public log
  loss 0.6073, rank 20. Model:
  `models/final_ensemble_cleanroom_v06.joblib`; package:
  `submissions/builds/trace_ace_cleanroom_v06.zip` (SHA-256
  `e55f1c7e2b969e88a042389ece54e05b04cd3e09e9538c3073c651c2af88b6b1`).
- v07 (bge-large semantic encoder) was built and packaged successfully but
  regressed on robust CV vs v06 — not submitted; kept as a record only.
- v08 (knowledge-graph-inspired dense features) extraction is done; CV
  ablation against v06 is the next step (`scripts/graph_ablation_cv.py`).

## Project layout

```text
trace-the-ace/
├── data/
│   ├── source/          # Immutable downloaded originals and transcript ZIP
│   ├── raw/             # Canonical extracted training inputs
│   ├── interim/         # Rebuildable intermediate tables and caches
│   └── processed/       # Final model-ready datasets
├── assets/               # Pinned offline model weights (ignored by Git)
├── configs/              # Resource and license registry
├── docs/
│   ├── competition/     # Competition rules, overview, and runtime contract
│   └── history/         # Imported prior-project notes; not current repo state
├── models/              # Trained artifacts (ignored by Git)
├── notebooks/           # Exploratory analysis
├── scripts/             # Reproducible cache/train/package commands
├── src/                 # Inspection, training, and inference code
├── submissions/         # Built submission archives and local outputs
├── submission_template/ # Root-level runtime entrypoint and notices
├── tests/                # Fast deterministic unit/integration tests
├── vendor/               # Pinned official runtime checkout
├── requirements.txt
└── venv/
```

See [`data/README.md`](data/README.md) for data-layer rules and [`docs/README.md`](docs/README.md) for the documentation map.

## Activate the environment

```bash
cd /opt/trace-the-ace
source venv/bin/activate
```

Confirm the active interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

Expected path:

```text
/opt/trace-the-ace/venv/bin/python
```

## Inspect the canonical raw tables

```bash
cd /opt/trace-the-ace
source venv/bin/activate
python src/inspect_data.py
```

The default inspection reads only the four top-level CSV tables in `data/raw/`. It summarizes nested directories without loading or printing the transcript corpus. Pass `--no-sample` to suppress table samples as well.

## Reproduce the modeling pipeline

All expensive stages are manifest-checked and reuse valid outputs:

```bash
python scripts/build_cache.py
python scripts/build_response_views.py
python scripts/build_sparse_store.py
python scripts/build_semantic_store.py
python scripts/run_sparse_cv.py
python scripts/run_semantic_cv.py
python scripts/evaluate_ensemble.py
python scripts/run_robust_cv.py
python scripts/train_final.py
python scripts/build_submission.py
```

Run the fast test suite at any point:

```bash
python -m pytest -q
```

The prepared ZIP—not a generated prediction CSV—is uploaded to the platform's
Code Jobs page. A platform smoke job is required before asking for confirmation
to consume the remaining Normal-submission quota.

## Verified input contract

| Input | Columns |
|---|---|
| Training features | `response_id`, `session_id`, `learning_objective_id`, `learning_objective` |
| Training labels | `response_id`, `is_correct` |
| Session transcripts | `session_id`, `utterance_id`, `role`, `content`, `timestamp` |
| Submission format | `response_id`, `probability` |

The training pipeline purges validation sessions, fits preprocessing only on
training folds, and saves every fitted component under `models/`. Prediction
code loads those saved artifacts and processes unseen test samples without
retraining.

## Data and runtime safeguards

- Never edit files in `data/source/`.
- Never commit or upload competition data.
- Never send transcript content to external APIs or third-party annotation services.
- Never derive inference features from aggregates or statistics across test samples.
- Never print test transcript text, objective text, or test-derived summaries in the competition runtime.
- JupyterLab is installed but is not started or publicly exposed by this project.
