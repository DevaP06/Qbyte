"""Unit + integration tests for src/blocking_token.py.

Run: python business_entity_resolution/tests/test_blocking_token.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from blocking_token import (  # noqa: E402
    _topk_per_row,
    build_token_index,
    run_blocking,
    select_blocking_keys,
)


def test_build_token_index_excludes_stopwords():
    # "pvt"/"ltd" (legal-suffix stopwords) must never become blocking-key
    # columns, even though they're present in the raw token stream.
    tokens = pa.array(["abc", "pvt", "ltd", "abc", "xyz"])
    row_ids = np.array([0, 0, 0, 1, 1])
    index = build_token_index(tokens, row_ids, n_rows=2, exclude=frozenset({"pvt", "ltd"}))
    assert set(index.key_to_col) == {"abc", "xyz"}
    assert index.matrix.shape == (2, 2)
    assert index.matrix[0, index.key_to_col["abc"]] == 1.0
    assert index.matrix[1, index.key_to_col["xyz"]] == 1.0


def test_build_token_index_duplicate_token_counts_once():
    tokens = pa.array(["abc", "abc", "abc"])
    row_ids = np.array([0, 0, 0])
    index = build_token_index(tokens, row_ids, n_rows=1)
    assert index.matrix.nnz == 1  # not 3 -- presence, not count


def test_select_blocking_keys_layer1_global_drop_no_fallback():
    # Column 0 is "too common" (df=100 > cap=50) and must be dropped
    # everywhere, with NO fallback to the row's full uncapped set even
    # though that would be this row's only token.
    query = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.float32))
    target_df = np.array([100, 5])
    selected = select_blocking_keys(query, target_df, max_document_frequency=50, keys_per_row=10)
    assert selected[0].nnz == 0  # row 0's only token was capped -> zero keys, not a fallback
    assert selected[1, 1] == 1.0


def test_select_blocking_keys_layer2_keeps_rarest_n():
    # Three surviving columns (all under the cap); keep only the 2 rarest.
    query = sp.csr_matrix(np.array([[1, 1, 1]], dtype=np.float32))
    target_df = np.array([50, 5, 20])  # col 1 rarest, then col 2, then col 0
    selected = select_blocking_keys(query, target_df, max_document_frequency=1000, keys_per_row=2)
    kept = set(selected.indices.tolist())
    assert kept == {1, 2}


def test_topk_per_row_deterministic_tie_break():
    scored = sp.csr_matrix(np.array([[0.5, 0.5, 0.9, 0.0]], dtype=np.float32))
    rows, cols, scores = _topk_per_row(scored, k=2)
    assert list(rows) == [0, 0]
    assert list(cols) == [2, 0]  # highest score first; ties broken by lower column index
    assert list(scores) == [0.9, 0.5]


def test_topk_per_row_empty_row():
    scored = sp.csr_matrix((2, 3), dtype=np.float32)
    rows, cols, scores = _topk_per_row(scored, k=5)
    assert len(rows) == 0 and len(cols) == 0 and len(scores) == 0


def _write_source(dir_path: Path, name: str, rows: list) -> None:
    pd.DataFrame(rows).to_parquet(dir_path / f"{name}.parquet", index=False)


def _build_synthetic_train(processed_dir: Path) -> pd.DataFrame:
    train_dir = processed_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    source1 = [
        {"entity_id": "S1-1", "country": "US", "name_tokens": ["abc", "retail"],
         "address_normalized": "123 main st springfield il"},
        {"entity_id": "S1-2", "country": "US", "name_tokens": ["xyz", "corp"],
         "address_normalized": "456 oak ave columbus oh"},
        {"entity_id": "S1-3", "country": "India", "name_tokens": ["abc", "pvt", "ltd"],
         "address_normalized": "mumbai maharashtra"},
        {"entity_id": "S1-4", "country": "US", "name_tokens": ["lonely", "singleton"],
         "address_normalized": "999 solo way loneville wy"},
    ]
    source2 = [
        {"entity_id": "S2-1", "country": "US", "name_tokens": ["abc", "retail", "inc"],
         "address_normalized": "123 main street springfield illinois"},
        {"entity_id": "S2-2", "country": "US", "name_tokens": ["unrelated", "business"],
         "address_normalized": "111 nowhere rd nowhere tx"},
        {"entity_id": "S2-3", "country": "India", "name_tokens": ["abc", "pvt", "ltd"],
         "address_normalized": "mumbai maharashtra india"},
    ]
    source3 = [
        {"entity_id": "S3-1", "country": "US", "name_tokens": ["xyz", "corporation"],
         "address_normalized": "456 oak avenue columbus ohio"},
        {"entity_id": "S3-2", "country": "US", "name_tokens": ["completely", "different"],
         "address_normalized": "1 elsewhere blvd anywhere ca"},
        # A second India target row so the India partition's target side has
        # n_docs > 1 -- with only one target document, idf = ln(n_docs/df)
        # is exactly ln(1/1) = 0 for every present token, zeroing every
        # score outright regardless of real overlap. Real country partitions
        # (millions of target rows) never hit this; it's purely a synthetic
        # test-scale artifact that would otherwise mask a genuine match.
        {"entity_id": "S3-3", "country": "India", "name_tokens": ["foo", "bar"],
         "address_normalized": "delhi india"},
    ]
    _write_source(train_dir, "source1", source1)
    _write_source(train_dir, "source2", source2)
    _write_source(train_dir, "source3", source3)

    return pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
            "matched_id": ["S2-1", "S3-1", "S2-3", None],
        }
    ).dropna()


def test_run_blocking_end_to_end_on_synthetic_corpus():
    """Exercises the whole wiring (country partitioning, two-layer key
    selection, batched scoring, id resolution, GT diagnostics) end-to-end on
    a tiny, hand-built dataset with known expected matches."""
    with tempfile.TemporaryDirectory() as tmp:
        processed_dir = Path(tmp)
        ground_truth = _build_synthetic_train(processed_dir)

        original_processed_dir = config.DATA_PROCESSED_DIR
        config.DATA_PROCESSED_DIR = processed_dir
        try:
            candidates, diagnostics = run_blocking(
                "train",
                max_document_frequency=1000,
                keys_per_side=10,
                max_candidates_per_entity=5,
                ground_truth=ground_truth,
            )
        finally:
            config.DATA_PROCESSED_DIR = original_processed_dir

    assert set(candidates.columns) == {"source1_entity_id", "candidate_entity_id", "score", "country"}
    assert (candidates["score"] > 0).all(), "every returned candidate must have shared at least one key"

    by_country = {d.country: d for d in diagnostics}
    assert set(by_country) == {"US", "India"}
    assert by_country["US"].n_query == 3  # S1-1, S1-2, S1-4
    assert by_country["India"].n_query == 1  # S1-3

    got = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    for pair in (("S1-1", "S2-1"), ("S1-2", "S3-1"), ("S1-3", "S2-3")):
        assert pair in got, f"expected match {pair} missing from candidates: {got}"

    assert ("S1-1", "S2-2") not in got  # no shared name/address key -- must not appear
    assert not any(s1 == "S1-4" for s1 in candidates["source1_entity_id"])  # true singleton -> zero candidates

    assert by_country["US"].n_gt_pairs == 2
    assert by_country["US"].recall_ceiling == 1.0
    assert by_country["US"].recall_at_k == 1.0
    assert by_country["India"].n_gt_pairs == 1
    assert by_country["India"].recall_ceiling == 1.0
    assert by_country["India"].recall_at_k == 1.0


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
