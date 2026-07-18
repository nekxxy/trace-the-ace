# Verified data inventory

Audit date: 2026-07-17 UTC

## Source downloads

| File | Rows/files | SHA-256 |
|---|---:|---|
| `train_features_TMQTWsB.csv` | 35,072 rows | `71bea3abb76a1cff5e1eaa75b9cbcfaf26d0419f6274b83a199ed520047a5063` |
| `train_labels_44ujmj2.csv` | 35,072 rows | `d98ee4389e5cde3f66d6d15b7b574261024a80e405958eca333d3c1921fd65b9` |
| `submission_format_5muR4s3.csv` | 100 rows | `ec4a4b1debdf05bce37c95bb3884c058c8807a8b12e1bd7d59f03ddcca560e6e` |
| `submission_format_ZQLcKx7.csv` | 10,508 rows | `957bc8b1fe9a69f74ee1bdc63edcc389810b191247645558396ea437b26ac0fa` |
| `train_transcripts.zip` | 22,821 CSV files | `e685b85b04694e130c25b17d09cdd1892fbda5e9fa685e98b2300114b915aa2d` |

## Verified schemas

- Features: `response_id`, `session_id`, `learning_objective_id`, `learning_objective`
- Labels: `response_id`, `is_correct`
- Transcripts: `session_id`, `utterance_id`, `role`, `content`, `timestamp`
- Submission format: `response_id`, `probability`

## Integrity summary

- 35,072 unique training responses join one-to-one between features and labels.
- 22,821 unique feature sessions have exactly one matching transcript file.
- The transcripts contain 6,139,854 utterances and 416,437,335 content characters.
- All transcript files share the same schema.
- No missing core cells, orphan sessions, missing sessions, malformed timestamps, invalid utterance IDs, duplicate source rows, or unreadable transcript files were found.
- Canonical files under `data/raw/` were byte-identical to the retained tabular originals at the time of the audit.
