# Implementation decisions

## Submission contract

- Official runtime source: `drivendataorg/tutoring-outcomes-runtime` commit
  `ea9a81755e101b8036e386430c3a2f3d7c655f2e`.
- Python 3.12; NumPy 2.2.6; pandas 3.0.3; scikit-learn 1.8.0;
  SciPy 1.17.1; PyArrow 24.0.0; joblib 1.5.3.
- The ZIP contains root-level `main.py`, a trained artifact, inference source,
  and every offline model asset. It creates root-level `submission.csv`.
- Runtime inference performs no fitting, network access, cross-test aggregation,
  pseudo-labeling, or data-derived logging.
- Fitting helpers live only in training modules that are excluded from the ZIP;
  packaged runtime modules contain transforms, artifact validation, and prediction only.

## Clean-room reconstruction boundary

The imported project history records a prior rank-16 ensemble and its results,
but its source, caches, trained artifact, and ZIP are not present. Exact
byte-for-byte reconstruction is therefore impossible. This project records
all recovered facts separately from new defaults.

Recovered design:

- objective-disjoint five-fold validation with validation-session purge;
- full-transcript hashed linear component (`alpha=3e-5`);
- role/objective/behavior linear component;
- BGE-small objective-context semantic interactions with logistic `C=0.1`;
- fixed probability blend `25% / 25% / 50%`.

New, explicit defaults:

- word `(1, 2)` hashing, nonnegative signs, float32, separate namespaces;
- `2^19` full, `2^18` tutor/student, and `2^16` objective dimensions;
- incremental averaged logistic SGD, four deterministic passes;
- 81 fixed per-response behavior, retrieval, and trajectory features;
- semantic interaction layout: context, objective, absolute difference,
  elementwise product, cosine similarity, and dense controls.

These choices must earn promotion through local OOF evidence and cannot be
described as the exact prior implementation.

## Validation policy

1. Primary metric is log loss; lower is better. AUROC, Brier, and ECE10 are
   diagnostics.
2. No response-row random split is permitted.
3. Primary stress split holds out complete objective IDs and purges all
   training rows from validation sessions.
4. Repeated semantic-family splits are the robustness gate for new model
   families.
5. Fixed ensemble weights are not retuned for a negligible local gain.
6. Promotion requires ensemble OOF log loss to be strictly lower than the
   leakage-safe fold-prior baseline and every individual component in the
   primary protocol and all four repeated semantic-family protocols.
7. The exact final ZIP must pass batch-invariant local inference and an
   official platform smoke job before a quota-consuming Normal submission.

## Artifact provenance

- Session caches, response views, sparse stores, and semantic stores bind both
  input hashes and the implementation files that determine their bytes.
- CV checkpoints bind the store manifest, configuration, protocol, exact fold
  masks, and trainer source hash; incompatible checkpoints fail closed.
- The final artifact records response, store, offline-asset, training-source,
  and packaged-runtime source hashes.
- The pinned BGE configuration uses the canonical repository ID rather than an
  upstream machine-local cache path; model weights and upstream revision are
  unchanged.

## Data and licensing

- Competition transcripts remain on this VPS and are excluded from Git and
  submission archives.
- No external training dataset is used.
- `BAAI/bge-small-en-v1.5` is pinned to revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` under the MIT license.
- Non-commercial and share-alike datasets are excluded from this build.
