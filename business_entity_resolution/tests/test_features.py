"""Unit + integration tests for src/features.py.

Run: python business_entity_resolution/tests/test_features.py
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from features import (  # noqa: E402
    EMBEDDING_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_feature_corpus,
    feature_columns_of,
    name_initials_and_compact,
    street_core,
    chunk_bounds,
    group_starts,
    iter_feature_chunks,
)
from io_utils import clean_dataframe, cleaned_chunk_to_table  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402


def _raw(entity_id, name, address, country):
    return {"entity_id": entity_id, "business_name": name, "business_address": address, "country": country}


SOURCE1 = [
    _raw("S1-1", "Abc Retail Pvt Ltd", "12 MG Road, Suite 100, Bengaluru, Karnataka", "India"),
    _raw("S1-2", "Blue Harbor Cafe", "1795 Westchester Drive, High Point, NC", "US"),
    _raw("S1-3", "Boulangerie Dupont", "4 Rue de la Paix, Paris, Ile-de-France", "France"),
]
SOURCE2 = [
    _raw("S2-1", "ABC Retail Private Limited", "12 MG Rd, Suite 100, Bengaluru, Karnataka", "India"),
    _raw("S2-2", "Abc Retail", "12 MG Road, Suite 101, Bengaluru, Karnataka", "India"),
    _raw("S2-3", "Blue Harbor Cafe", None, "US"),
    _raw("S2-4", "Boulangerie Dupont", "4 Rue de la Paix, Paris, Ile-de-France", "France"),
]
SOURCE3 = [
    _raw("S3-1", "Blue Harbour Café", "1795 Westchester Dr, High Point, NC", "US"),
    _raw("S3-2", "Red Rock Diner", "22 Elm Street, Tulsa, OK", "US"),
]

CANDIDATES = pd.DataFrame(
    [
        ("S1-1", "S2-1", 9.0, "India"),
        ("S1-1", "S2-2", 7.0, "India"),
        ("S1-2", "S3-1", 8.0, "US"),
        ("S1-2", "S2-3", 5.0, "US"),
        ("S1-2", "S3-2", 1.0, "US"),
        ("S1-3", "S2-4", 6.0, "France"),
    ],
    columns=["source1_entity_id", "candidate_entity_id", "score", "country"],
)


def _write_cleaned_split(processed_dir: Path) -> None:
    split_dir = processed_dir / "test"
    split_dir.mkdir(parents=True)
    for name, rows in (("source1", SOURCE1), ("source2", SOURCE2), ("source3", SOURCE3)):
        cleaned = clean_dataframe(pd.DataFrame(rows))
        pq.write_table(cleaned_chunk_to_table(cleaned), split_dir / f"{name}.parquet")


def _featurize(chunk_pairs: int, candidates: pd.DataFrame = CANDIDATES) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        original = config.DATA_PROCESSED_DIR
        config.DATA_PROCESSED_DIR = Path(tmp)
        try:
            _write_cleaned_split(Path(tmp))
            corpus = build_feature_corpus("test")
        finally:
            config.DATA_PROCESSED_DIR = original
    chunks = list(iter_feature_chunks(corpus, candidates, chunk_pairs=chunk_pairs))
    return pd.concat(chunks, ignore_index=True).set_index(["source1_entity_id", "candidate_entity_id"])


# A candidate_generation.py-style union: S3-2 found only by embeddings (token
# score 0); S2-3 found only by tokens (no embedding rank).
UNION_CANDIDATES = CANDIDATES.assign(
    score=[9.0, 7.0, 8.0, 5.0, 0.0, 6.0],
    emb_score=[0.95, 0.90, 0.92, 0.80, 0.30, 0.99],
    emb_rank=[0.0, 1.0, 0.0, np.nan, 1.0, 0.0],
)


FEATURES = None


def _features() -> pd.DataFrame:
    global FEATURES
    if FEATURES is None:
        FEATURES = _featurize(chunk_pairs=1_000)
    return FEATURES


def test_output_schema_and_row_count():
    f = _features()
    assert len(f) == len(CANDIDATES)
    assert list(f.columns) == ["country"] + FEATURE_COLUMNS
    assert "country" not in FEATURE_COLUMNS  # passthrough for diagnostics, never a model feature


def test_identical_records_score_perfectly():
    row = _features().loc[("S1-3", "S2-4")]
    for col in ("name_ratio", "name_token_set_ratio", "name_tok_jaccard", "name_char3_dice", "addr_ratio"):
        assert math.isclose(row[col], 1.0, abs_tol=1e-6), (col, row[col])
    for col in ("name_tok_idf_cos", "name_char3_idf_cos", "addr_tok_idf_cos"):
        assert math.isclose(row[col], 1.0, abs_tol=1e-5), (col, row[col])


def test_legal_suffix_variants_match_as_tokens():
    # "Pvt Ltd" vs "Private Limited" canonicalize to the same tokens in normalize.py.
    row = _features().loc[("S1-1", "S2-1")]
    assert math.isclose(row["name_tok_jaccard"], 1.0), row["name_tok_jaccard"]


def test_address_digit_conflict_suite_100_vs_101():
    f = _features()
    same = f.loc[("S1-1", "S2-1")]
    diff = f.loc[("S1-1", "S2-2")]
    assert same["addr_digit_conflict"] == 0.0 and same["addr_digit_match"] == 1.0
    # {12, 100} vs {12, 101}: shares "12", so not a full conflict -- but a lower digit Jaccard.
    assert diff["addr_digit_conflict"] == 0.0
    assert diff["addr_digit_jaccard"] < same["addr_digit_jaccard"]
    # {1795} vs {22}: both sides numbered, nothing shared -> conflict.
    assert f.loc[("S1-2", "S3-2"), "addr_digit_conflict"] == 1.0


def test_missing_address_gives_nan_not_zero_or_perfect():
    row = _features().loc[("S1-2", "S2-3")]
    assert row["addr_both_present"] == 0.0
    for col in ("addr_ratio", "addr_token_set_ratio", "addr_head_ratio", "addr_tok_jaccard",
                "addr_tok_idf_cos", "addr_digit_conflict"):
        assert np.isnan(row[col]), (col, row[col])
    assert math.isclose(row["name_ratio"], 1.0)  # name features unaffected


def test_context_features():
    f = _features()
    us = f.loc["S1-2"]
    assert list(us["block_rank"].sort_values()) == [0.0, 1.0, 2.0]
    assert us.loc["S3-1", "block_rank"] == 0.0  # highest blocking score
    assert math.isclose(us.loc["S3-1", "block_score_rel"], 1.0)
    assert (us["n_candidates"] == 3).all()
    assert us.loc["S3-2", "name_tok_idf_cos_rank"] == 2.0  # unrelated diner ranks last
    assert us["name_tok_idf_cos_gap"].min() == 0.0  # the best candidate has zero gap
    assert us.loc["S3-2", "name_tok_idf_cos_gap"] > 0.5


def test_ascii_view_ignores_diacritics():
    # "blue harbor cafe" vs "blue harbour café": the é only costs points on
    # the Unicode-preserving view.
    row = _features().loc[("S1-2", "S3-1")]
    assert row["name_ascii_token_set_ratio"] > row["name_token_set_ratio"]


def test_source_indicator():
    f = _features()
    assert f.loc[("S1-2", "S3-1"), "cand_is_source3"] == 1.0
    assert f.loc[("S1-2", "S2-3"), "cand_is_source3"] == 0.0


def test_chunking_never_splits_an_entity_and_is_deterministic():
    starts = group_starts(np.array([0, 0, 0, 1, 1, 2]))
    assert list(starts) == [0, 3, 5]
    # target=1 must still keep each whole group together.
    assert chunk_bounds(starts, 6, target=1) == [(0, 3), (3, 5), (5, 6)]
    assert chunk_bounds(starts, 6, target=100) == [(0, 6)]

    tiny = _featurize(chunk_pairs=1).sort_index()
    whole = _features().sort_index()
    pd.testing.assert_frame_equal(tiny, whole)


def test_embedding_features_from_union_candidates():
    f = _featurize(chunk_pairs=1_000, candidates=UNION_CANDIDATES)
    assert list(f.columns) == ["country"] + FEATURE_COLUMNS + EMBEDDING_FEATURE_COLUMNS
    us = f.loc["S1-2"]
    assert us.loc["S3-2", "from_token"] == 0.0 and us.loc["S3-1", "from_token"] == 1.0
    assert np.isnan(us.loc["S2-3", "emb_rank"])  # tokens only -> not an embedding neighbour
    assert math.isclose(us.loc["S3-1", "emb_cos"], 0.92, rel_tol=1e-6)
    assert us.loc["S3-1", "emb_cos_gap"] == 0.0 and us.loc["S3-1", "emb_cos_rank"] == 0.0
    assert math.isclose(us.loc["S3-2", "emb_cos_gap"], 0.62, abs_tol=1e-6)
    assert us.loc["S3-2", "block_rank"] == 2.0  # embedding-only pairs rank after every token pair


def test_street_core():
    assert street_core("12 rue des travailleurs, lille, hauts-de-france") == "travailleurs"
    assert street_core("12 r. de crimee, lille") == "crimee"
    assert street_core("il, chicago, 1840 blue island avenue") == "blue island"  # reordered address
    assert street_core("616 81st street, chicago, il") == "81st"  # ordinals kept, house numbers dropped
    assert street_core("9628c spyglass drive, oregon") == "spyglass"
    assert street_core("plot no. 20, jaipur") == ""  # nothing identifying left
    assert street_core("") == ""


def test_name_acronyms():
    assert name_initials_and_compact(["art", "forces", "comite", "eurl"]) == ("afc", "artforcescomite")
    assert name_initials_and_compact(["afc"]) == ("a", "afc")
    assert name_initials_and_compact(["pvt", "ltd"]) == ("", "")


def test_street_core_ratio_and_acronym_in_features():
    f = _features()
    same = f.loc[("S1-3", "S2-4")]  # identical French record
    assert same["street_core_ratio"] == 1.0 and same["name_acronym_match"] == 0.0
    assert np.isnan(f.loc[("S1-2", "S2-3"), "street_core_ratio"])  # candidate has no address


def test_feature_columns_of_reads_the_schema():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _featurize(chunk_pairs=1_000, candidates=UNION_CANDIDATES).reset_index().to_parquet(d / "part-00000.parquet")
        assert feature_columns_of(d) == FEATURE_COLUMNS + EMBEDDING_FEATURE_COLUMNS
        _features().reset_index().to_parquet(d / "part-00000.parquet")
        assert feature_columns_of(d) == FEATURE_COLUMNS


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
