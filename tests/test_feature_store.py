from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trace_ace.cache import build_cache
from trace_ace.config import ModelConfig
from trace_ace.feature_store import build_response_views, dense_columns
from trace_ace.features import FeatureConfig
from trace_ace.sparse_store import build_sparse_store, iter_sparse_parts, load_dense_by_row


def _raw_fixture(root: Path) -> tuple[Path, Path]:
    raw = root / "raw"
    transcripts = raw / "train_transcripts"
    transcripts.mkdir(parents=True)
    pd.DataFrame(
        [
            ["r1", "s1", "o1", "add whole numbers"],
            ["r2", "s1", "o2", "explain reasoning"],
            ["r3", "s2", "o1", "add whole numbers"],
            ["r4", "s2", "o2", "explain reasoning"],
        ],
        columns=["response_id", "session_id", "learning_objective_id", "learning_objective"],
    ).to_csv(raw / "train_features_fixture.csv", index=False)
    pd.DataFrame(
        [["r1", 1], ["r2", 0], ["r3", 0], ["r4", 1]],
        columns=["response_id", "is_correct"],
    ).to_csv(raw / "train_labels_fixture.csv", index=False)
    columns = ["session_id", "utterance_id", "role", "content", "timestamp"]
    pd.DataFrame(
        [
            ["s1", 1, "tutor", "How do we add the numbers?", "00:00:01"],
            ["s1", 2, "student", "I add each place because they align.", "00:00:03"],
            ["s1", 3, "tutor", "Correct, good reasoning.", "00:00:05"],
        ],
        columns=columns,
    ).to_csv(transcripts / "s1.csv", index=False)
    pd.DataFrame(
        [
            ["s2", 1, "tutor", "Add these whole numbers.", "00:00:01"],
            ["s2", 2, "student", "I am not sure.", "00:00:02"],
            ["s2", 3, "tutor", "Try again and explain why.", "00:00:04"],
        ],
        columns=columns,
    ).to_csv(transcripts / "s2.csv", index=False)
    return raw, transcripts


def test_response_and_sparse_stores_integrate(tmp_path: Path) -> None:
    raw, transcripts = _raw_fixture(tmp_path)
    cache = build_cache(raw, tmp_path / "cache", batch_size=1)
    response_views = build_response_views(
        responses_path=cache.responses_path,
        session_views_path=cache.session_views_path,
        transcripts_dir=transcripts,
        output_dir=tmp_path / "response_views",
        feature_config=FeatureConfig(context_char_budget=256),
        batch_size=2,
    )
    assert response_views.response_count == 4
    frame = pq.read_table(response_views.path).to_pandas().sort_values("row_id")
    assert frame["row_id"].tolist() == [0, 1, 2, 3]
    assert frame["objective_context"].str.len().gt(0).all()
    assert np.isfinite(frame.loc[:, dense_columns()].to_numpy()).all()

    config = ModelConfig(
        full_hash_features=2**9,
        role_hash_features=2**8,
        objective_hash_features=2**7,
        sparse_epochs=1,
    )
    sparse_result = build_sparse_store(
        response_views.path,
        tmp_path / "sparse",
        config=config,
        batch_size=2,
    )
    assert sparse_result.part_count == 2
    parts = list(iter_sparse_parts(sparse_result.path))
    assert sum(len(part.row_ids) for part in parts) == 4
    assert parts[0].full.shape[1] == 2**9 + 2**7
    assert parts[0].role.shape[1] == 2 * 2**8 + 2**7
    target, dense = load_dense_by_row(sparse_result.path)
    assert target.tolist() == [1, 0, 0, 1]
    assert dense.shape == (4, len(dense_columns()))

    reused = build_sparse_store(
        response_views.path,
        sparse_result.path,
        config=config,
        batch_size=2,
    )
    assert reused.status == "reused"
