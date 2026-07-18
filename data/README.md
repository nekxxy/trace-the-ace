# Data layout

Competition data is private, remains local to this project, and is excluded from Git.

- `source/`: immutable files exactly as downloaded or uploaded. This includes the original transcript ZIP and original CSV tables. Never edit these files.
- `raw/`: canonical working copies of the feature, label, submission-format, and extracted transcript inputs. Load training data from here.
- `interim/`: rebuildable caches and intermediate representations derived from `raw/`.
- `processed/`: finalized model-ready datasets produced by deterministic preprocessing.

Every derived artifact should be reproducible from `source/` or `raw/` plus versioned code. Do not place notebooks, trained models, or submissions in this directory.
