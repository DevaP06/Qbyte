"""Pairwise feature engineering over blocking candidates (plan.md build-order step 3).

Input: data/processed/candidates/<split>/token_blocking.parquet (blocking_token.py)
plus the cleaned source Parquet (clean_all.py) -- never re-normalizes text.
Output: one row per candidate pair, `FEATURE_COLUMNS` plus the two ids and
`country`, written as partitioned Parquet to
data/processed/features/<split>/part-NNNNN.parquet. Labels are NOT joined here
(that is labels.py's job, and needs train_ground_truth.tsv); train and test go
through the identical code path.

Three kinds of feature, all country-agnostic (no country one-hot, no
per-country table -- France is handled by the same code as US/India):

  * String-level similarity (rapidfuzz `cpdist`: C, multi-threaded, measured
    at ~0.05-0.1s per scorer per 1M pairs) on the Unicode-preserving text, so
    Devanagari/Kannada names still compare against each other.
  * Set-level similarity from sparse token incidence matrices, computed as a
    row-wise product of the gathered S1 rows and candidate rows -- one
    elementwise product per token space yields shared-count, shared-IDF, and
    shared-IDF^2 at once (Jaccard, IDF cosine, IDF containment). IDF is
    computed per country over that split's own S1+S2+S3 rows: the same "group
    by whatever string is in `country`" rule blocking uses, so France gets its
    own IDF at test time with no special case.
  * Per-S1 context: each candidate's gap to / rank against the best candidate
    of the same S1 entity. F_0.5 is scored per S1 entity, so "is this the best
    of this entity's options" matters as much as the absolute similarity.

Deviations from plan.md's Task 2 text:
  - Monge-Elkan is skipped: it needs a Python-level per-token-pair loop, which
    is intractable at ~88M train pairs; `token_set_ratio` + IDF containment
    cover the same "partial token overlap" failure mode in C.
  - No all-NaN embedding-cosine placeholder column: it carries no signal for
    the CPU baseline. It gets added as a real column once embeddings.py exists.
  - The raw blocking score is not a feature, only its within-entity rank and
    ratio to the entity's best: its absolute scale depends on the size of the
    country partition (blocking's IDF), which differs for France.
  - `country` is a passthrough column for per-country diagnostics only, never
    a feature (plan.md: constant within a pair, and a one-hot breaks France).
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer

import config
from blocking_token import _flatten_token_list_column, _tokenize_address_column, build_token_index
from clean_all import ProgressBar
from io_utils import read_cleaned_table

ID_COLUMNS = ["source1_entity_id", "candidate_entity_id"]

FEATURE_COLUMNS = [
    # name: string-level
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_ascii_token_set_ratio",
    # name: set-level
    "name_tok_jaccard",
    "name_tok_idf_cos",
    "name_tok_idf_containment_s1",
    "name_tok_idf_containment_cand",
    "name_char3_dice",
    "name_char3_idf_cos",
    # name: digits ("Suite 100 vs Suite 101"), length, script
    "name_digit_match",
    "name_digit_conflict",
    "name_len_s1",
    "name_len_cand",
    "name_len_ratio",
    "name_nonascii_frac_s1",
    "name_nonascii_frac_cand",
    # address: string-level, incl. positional head/middle/tail
    "addr_ratio",
    "addr_token_set_ratio",
    "addr_head_ratio",
    "addr_middle_token_set_ratio",
    "addr_tail_ratio",
    # address: set-level
    "addr_tok_jaccard",
    "addr_tok_idf_cos",
    "addr_tok_idf_containment_s1",
    "addr_tok_idf_containment_cand",
    # address: digits (house/unit numbers), presence, length
    "addr_digit_match",
    "addr_digit_conflict",
    "addr_digit_jaccard",
    "addr_both_present",
    "addr_len_ratio",
    # metadata
    "cand_is_source3",
    # blocking
    "block_rank",
    "block_score_rel",
    "n_candidates",
    # per-S1 context
    "name_tok_idf_cos_gap",
    "name_token_set_ratio_gap",
    "name_char3_idf_cos_gap",
    "addr_tok_idf_cos_gap",
    "name_tok_idf_cos_rank",
    "addr_tok_idf_cos_rank",
]

# Added when the candidates come from candidate_generation.py (token blocking
# union embedding retrieval) and so carry an embedding cosine for every pair.
EMBEDDING_FEATURE_COLUMNS = [
    "emb_cos",  # cosine of the two records' embeddings
    "emb_rank",  # rank among the S1's embedding neighbours; NaN = not retrieved by embeddings
    "emb_cos_gap",  # best emb_cos among this S1's candidates minus this one
    "emb_cos_rank",  # rank by emb_cos among this S1's candidates
    "from_token",  # 1 if token blocking retrieved the pair (score > 0)
]


def feature_columns_of(features_dir) -> list:
    """The model feature columns of a features dataset, from its schema, in
    order -- the base set, plus the embedding set if it was built from a
    candidate union. Training and prediction both read the list from here."""
    import pyarrow.parquet as pq

    first = next(iter(sorted(features_dir.glob("part-*.parquet"))), None)
    if first is None:
        raise FileNotFoundError(f"no feature parts in {features_dir} -- run features.py first")
    present = set(pq.read_schema(first).names)
    return [c for c in FEATURE_COLUMNS + EMBEDDING_FEATURE_COLUMNS if c in present]


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, NaN wherever den == 0 (an undefined similarity, not a zero one)."""
    out = np.full(len(num), np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _row_sums(m: sp.csr_matrix) -> np.ndarray:
    return np.asarray(m.sum(axis=1), dtype=np.float32).ravel()


def _take_strings(arr: pa.Array, idx: np.ndarray) -> np.ndarray:
    # to_numpy (object array) measured ~6x faster than to_pylist for 1M
    # strings, and rapidfuzz's cpdist accepts it directly.
    return arr.take(pa.array(idx)).to_numpy(zero_copy_only=False)


# --------------------------------------------------------------------------
# Token spaces: incidence matrices + per-country IDF
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenSpace:
    """One token vocabulary (e.g. name words) over every row of a split.

    `binary` is the 0/1 (row x key) incidence; `weighted` has the same
    sparsity with each entry replaced by that key's IDF in the row's own
    country. `size`, `idf_sum`, `idf_norm` are per-row |A|, sum(idf),
    sqrt(sum(idf^2)), precomputed once so pair features need only one
    elementwise product per chunk.
    """

    binary: sp.csr_matrix
    weighted: sp.csr_matrix
    size: np.ndarray
    idf_sum: np.ndarray
    idf_norm: np.ndarray


def _country_idf_weighted(binary: sp.csr_matrix, country_codes: np.ndarray, n_countries: int) -> sp.csr_matrix:
    """Replace each 1 in `binary` by the key's smoothed IDF within that row's
    country: idf = ln((N_c + 1) / (df_c + 1)) + 1, always >= 1, so an
    elementwise product never produces an explicit zero that scipy would
    prune (shared-token counts below rely on that)."""
    n_rows, n_cols = binary.shape
    row_of_nnz = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(binary.indptr))
    country_of_nnz = country_codes[row_of_nnz]
    del row_of_nnz
    weights = np.empty(binary.nnz, dtype=np.float32)
    for c in range(n_countries):
        in_c = country_of_nnz == c
        cols = binary.indices[in_c]
        df = np.bincount(cols, minlength=n_cols)
        n_docs = int((country_codes == c).sum())
        idf = (np.log((n_docs + 1) / (df + 1)) + 1).astype(np.float32)
        weights[in_c] = idf[cols]
    return sp.csr_matrix((weights, binary.indices, binary.indptr), shape=binary.shape)


def build_token_space(binary: sp.csr_matrix, country_codes: np.ndarray, n_countries: int) -> TokenSpace:
    binary = binary.tocsr()
    weighted = _country_idf_weighted(binary, country_codes, n_countries)
    squared = sp.csr_matrix((weighted.data**2, weighted.indices, weighted.indptr), shape=weighted.shape)
    return TokenSpace(
        binary=binary,
        weighted=weighted,
        size=np.diff(binary.indptr).astype(np.float32),
        idf_sum=_row_sums(weighted),
        idf_norm=np.sqrt(_row_sums(squared)),
    )


def _digit_columns_only(binary: sp.csr_matrix, key_to_col: dict) -> sp.csr_matrix:
    """Restrict an incidence matrix to its all-digit keys ("100", "1795")."""
    is_digit_col = np.zeros(binary.shape[1], dtype=bool)
    is_digit_col[[col for tok, col in key_to_col.items() if tok.isdigit()]] = True
    out = binary.copy()
    out.data = out.data * is_digit_col[out.indices]
    out.eliminate_zeros()
    return out


def _char_trigram_matrix(texts: pa.Array, batch_rows: int = 500_000) -> sp.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        n_features=config.FEATURE_CHAR_NGRAM_HASH_SIZE,
        alternate_sign=False,
        norm=None,
        binary=True,
        lowercase=False,  # already lowercased by normalize.py
        dtype=np.float32,
    )
    parts = [
        vectorizer.transform(texts.slice(s, batch_rows).to_pylist())
        for s in range(0, len(texts), batch_rows)
    ]
    return sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, config.FEATURE_CHAR_NGRAM_HASH_SIZE))


# --------------------------------------------------------------------------
# Per-split corpus: every S1/S2/S3 row, one shared row space
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureCorpus:
    """All rows of one split, source1 first then source2, source3 -- the same
    row layout as blocking_token.CorpusIndex. Text columns stay as compact
    Arrow arrays (nulls filled with "") and are materialized per chunk only."""

    entity_ids: np.ndarray
    countries: np.ndarray
    is_source3: np.ndarray
    address_present: np.ndarray
    name_text: pa.Array
    name_ascii: pa.Array
    name_ascii_len: np.ndarray
    name_len: np.ndarray
    name_nonascii_frac: np.ndarray
    addr_text: pa.Array
    addr_head: pa.Array
    addr_middle: pa.Array
    addr_middle_len: np.ndarray
    addr_tail: pa.Array
    addr_len: np.ndarray
    name_words: TokenSpace
    name_char3: TokenSpace
    addr_words: TokenSpace
    name_digits: sp.csr_matrix
    addr_digits: sp.csr_matrix


_CORPUS_COLUMNS = [
    "entity_id",
    "country",
    "name_original_normalized",
    "name_ascii_folded",
    "name_tokens",
    "address_present",
    "address_normalized",
    "address_head",
    "address_middle",
    "address_tail",
]


def _filled(table: pa.Table, column: str) -> pa.Array:
    return pc.fill_null(table.column(column).combine_chunks(), "")


def build_feature_corpus(split: str) -> FeatureCorpus:
    tables = [read_cleaned_table(split, s, columns=_CORPUS_COLUMNS) for s in ("source1", "source2", "source3")]
    table = pa.concat_tables(tables)
    n_rows = table.num_rows
    is_source3 = np.zeros(n_rows, dtype=bool)
    is_source3[tables[0].num_rows + tables[1].num_rows :] = True

    entity_ids = table.column("entity_id").to_numpy()
    countries = table.column("country").to_numpy()
    country_values, country_codes = np.unique(countries, return_inverse=True)
    country_codes = country_codes.astype(np.int8)
    n_countries = len(country_values)

    name_text = _filled(table, "name_original_normalized")
    name_ascii = _filled(table, "name_ascii_folded")
    name_len = pc.utf8_length(name_text).to_numpy().astype(np.float32)
    ascii_len = pc.utf8_length(name_ascii).to_numpy().astype(np.float32)
    # ascii_folded drops non-Latin characters but only strips diacritics from
    # Latin ones, so this is ~the share of the name that is non-Latin script.
    # Tells the model when name similarities are structurally unreliable
    # (Latin S1 vs Devanagari candidate) and address should carry the pair.
    name_nonascii_frac = np.clip(1.0 - _safe_div(ascii_len, name_len), 0.0, 1.0)

    addr_text = _filled(table, "address_normalized")
    addr_middle = pc.fill_null(pc.binary_join(table.column("address_middle").combine_chunks(), " "), "")

    name_flat, name_rows = _flatten_token_list_column(table.column("name_tokens").combine_chunks())
    name_index = build_token_index(name_flat, name_rows, n_rows)  # generic tokens kept -- IDF down-weights them
    addr_flat, addr_rows = _tokenize_address_column(addr_text.to_numpy(zero_copy_only=False))
    addr_index = build_token_index(addr_flat, addr_rows, n_rows)

    return FeatureCorpus(
        entity_ids=entity_ids,
        countries=countries,
        is_source3=is_source3,
        address_present=table.column("address_present").to_numpy().astype(bool),
        name_text=name_text,
        name_ascii=name_ascii,
        name_ascii_len=ascii_len,
        name_len=name_len,
        name_nonascii_frac=name_nonascii_frac,
        addr_text=addr_text,
        addr_head=_filled(table, "address_head"),
        addr_middle=addr_middle,
        addr_middle_len=pc.utf8_length(addr_middle).to_numpy().astype(np.float32),
        addr_tail=_filled(table, "address_tail"),
        addr_len=pc.utf8_length(addr_text).to_numpy().astype(np.float32),
        name_words=build_token_space(name_index.matrix, country_codes, n_countries),
        name_char3=build_token_space(_char_trigram_matrix(name_text), country_codes, n_countries),
        addr_words=build_token_space(addr_index.matrix, country_codes, n_countries),
        name_digits=_digit_columns_only(name_index.matrix, name_index.key_to_col),
        addr_digits=_digit_columns_only(addr_index.matrix, addr_index.key_to_col),
    )


# --------------------------------------------------------------------------
# Pair features
# --------------------------------------------------------------------------


def _set_similarities(space: TokenSpace, q: np.ndarray, t: np.ndarray) -> dict:
    # weighted[q] * binary[t]: one entry (= the key's IDF) per shared key.
    shared = space.weighted[q].multiply(space.binary[t]).tocsr()
    n_shared = np.diff(shared.indptr).astype(np.float32)
    shared_idf = _row_sums(shared)
    shared_idf_sq = _row_sums(shared.multiply(shared))
    size_q, size_t = space.size[q], space.size[t]
    return {
        "jaccard": _safe_div(n_shared, size_q + size_t - n_shared),
        "dice": _safe_div(2 * n_shared, size_q + size_t),
        "idf_cos": _safe_div(shared_idf_sq, space.idf_norm[q] * space.idf_norm[t]),
        "containment_s1": _safe_div(shared_idf, space.idf_sum[q]),
        "containment_cand": _safe_div(shared_idf, space.idf_sum[t]),
    }


def _digit_features(digits: sp.csr_matrix, q: np.ndarray, t: np.ndarray) -> dict:
    dq, dt = digits[q], digits[t]
    n_shared = np.diff(dq.multiply(dt).tocsr().indptr).astype(np.float32)
    size_q = np.diff(dq.indptr).astype(np.float32)
    size_t = np.diff(dt.indptr).astype(np.float32)
    return {
        "match": (n_shared > 0).astype(np.float32),
        # Both sides carry numbers and none agree: the "Suite 100 vs Suite
        # 101" false-merge case plan.md calls out.
        "conflict": ((size_q > 0) & (size_t > 0) & (n_shared == 0)).astype(np.float32),
        "jaccard": _safe_div(n_shared, size_q + size_t - n_shared),
    }


def _string_similarity(a: list, b: list, scorer, valid: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """rapidfuzz scorer over aligned pairs, scaled to [0, 1]; NaN where
    `valid` is False (rapidfuzz scores ratio("", "") as a perfect 100)."""
    scores = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale
    scores[~valid] = np.nan
    return scores


def compute_pair_features(corpus: FeatureCorpus, q: np.ndarray, t: np.ndarray) -> dict:
    """Every pair-local feature for aligned (S1 row, candidate row) arrays.
    Per-S1 context and blocking-rank features are added separately
    (`add_context_features`), since they need the whole entity's group."""
    f: dict = {}

    name_q, name_t = _take_strings(corpus.name_text, q), _take_strings(corpus.name_text, t)
    name_ok = (corpus.name_len[q] > 0) & (corpus.name_len[t] > 0)
    f["name_ratio"] = _string_similarity(name_q, name_t, fuzz.ratio, name_ok)
    f["name_partial_ratio"] = _string_similarity(name_q, name_t, fuzz.partial_ratio, name_ok)
    f["name_token_sort_ratio"] = _string_similarity(name_q, name_t, fuzz.token_sort_ratio, name_ok)
    f["name_token_set_ratio"] = _string_similarity(name_q, name_t, fuzz.token_set_ratio, name_ok)
    f["name_jaro_winkler"] = _string_similarity(
        name_q, name_t, JaroWinkler.normalized_similarity, name_ok, scale=1.0
    )
    del name_q, name_t

    # Diacritic-insensitive view ("frànce sàrl" vs "france sarl", seen in the
    # France test data). NaN when either side has no Latin text left, e.g. a
    # Devanagari-only name -- undefined there, not a mismatch.
    ascii_ok = (corpus.name_ascii_len[q] > 0) & (corpus.name_ascii_len[t] > 0)
    f["name_ascii_token_set_ratio"] = _string_similarity(
        _take_strings(corpus.name_ascii, q), _take_strings(corpus.name_ascii, t), fuzz.token_set_ratio, ascii_ok
    )

    words = _set_similarities(corpus.name_words, q, t)
    f["name_tok_jaccard"] = words["jaccard"]
    f["name_tok_idf_cos"] = words["idf_cos"]
    f["name_tok_idf_containment_s1"] = words["containment_s1"]
    f["name_tok_idf_containment_cand"] = words["containment_cand"]
    char3 = _set_similarities(corpus.name_char3, q, t)
    f["name_char3_dice"] = char3["dice"]
    f["name_char3_idf_cos"] = char3["idf_cos"]

    name_digits = _digit_features(corpus.name_digits, q, t)
    f["name_digit_match"] = name_digits["match"]
    f["name_digit_conflict"] = name_digits["conflict"]

    len_q, len_t = corpus.name_len[q], corpus.name_len[t]
    f["name_len_s1"] = len_q
    f["name_len_cand"] = len_t
    f["name_len_ratio"] = _safe_div(np.minimum(len_q, len_t), np.maximum(len_q, len_t))
    f["name_nonascii_frac_s1"] = corpus.name_nonascii_frac[q]
    f["name_nonascii_frac_cand"] = corpus.name_nonascii_frac[t]

    addr_ok = corpus.address_present[q] & corpus.address_present[t]
    addr_q, addr_t = _take_strings(corpus.addr_text, q), _take_strings(corpus.addr_text, t)
    f["addr_ratio"] = _string_similarity(addr_q, addr_t, fuzz.ratio, addr_ok)
    f["addr_token_set_ratio"] = _string_similarity(addr_q, addr_t, fuzz.token_set_ratio, addr_ok)
    del addr_q, addr_t
    f["addr_head_ratio"] = _string_similarity(
        _take_strings(corpus.addr_head, q), _take_strings(corpus.addr_head, t), fuzz.ratio, addr_ok
    )
    middle_ok = addr_ok & (corpus.addr_middle_len[q] > 0) & (corpus.addr_middle_len[t] > 0)
    f["addr_middle_token_set_ratio"] = _string_similarity(
        _take_strings(corpus.addr_middle, q), _take_strings(corpus.addr_middle, t), fuzz.token_set_ratio, middle_ok
    )
    f["addr_tail_ratio"] = _string_similarity(
        _take_strings(corpus.addr_tail, q), _take_strings(corpus.addr_tail, t), fuzz.ratio, addr_ok
    )

    addr_words = _set_similarities(corpus.addr_words, q, t)
    for key, col in (
        ("jaccard", "addr_tok_jaccard"),
        ("idf_cos", "addr_tok_idf_cos"),
        ("containment_s1", "addr_tok_idf_containment_s1"),
        ("containment_cand", "addr_tok_idf_containment_cand"),
    ):
        values = addr_words[key]
        values[~addr_ok] = np.nan
        f[col] = values

    addr_digits = _digit_features(corpus.addr_digits, q, t)
    for key in ("match", "conflict", "jaccard"):
        values = addr_digits[key]
        values[~addr_ok] = np.nan
        f[f"addr_digit_{key}"] = values

    f["addr_both_present"] = addr_ok.astype(np.float32)
    alen_q, alen_t = corpus.addr_len[q], corpus.addr_len[t]
    f["addr_len_ratio"] = _safe_div(np.minimum(alen_q, alen_t), np.maximum(alen_q, alen_t))

    f["cand_is_source3"] = corpus.is_source3[t].astype(np.float32)
    return f


# --------------------------------------------------------------------------
# Per-S1 context features (need the entity's whole candidate group)
# --------------------------------------------------------------------------


def group_starts(sorted_q: np.ndarray) -> np.ndarray:
    """Start offsets of each run of equal values in an already-grouped array."""
    if len(sorted_q) == 0:
        return np.array([], dtype=np.int64)
    return np.flatnonzero(np.r_[True, sorted_q[1:] != sorted_q[:-1]])


def _gap_to_group_max(values: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    group_max = np.fmax.reduceat(values, starts)  # fmax: ignores NaN unless the whole group is NaN
    return (np.repeat(group_max, sizes) - values).astype(np.float32)


def _rank_in_group(values: np.ndarray, starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """0 = best (highest) value within its group; NaN values rank last."""
    group_id = np.repeat(np.arange(len(starts)), sizes)
    order = np.lexsort((-values, group_id))
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values)) - np.repeat(starts, sizes)
    return ranks


def add_context_features(f: dict, sorted_q: np.ndarray, block_score: np.ndarray) -> None:
    """In place. Rows must be grouped by S1 and, within a group, ordered by
    descending blocking score (see `iter_feature_chunks`)."""
    starts = group_starts(sorted_q)
    sizes = np.diff(np.r_[starts, len(sorted_q)])
    f["block_rank"] = (np.arange(len(sorted_q)) - np.repeat(starts, sizes)).astype(np.float32)
    f["block_score_rel"] = _safe_div(block_score, np.repeat(np.maximum.reduceat(block_score, starts), sizes))
    f["n_candidates"] = np.repeat(sizes, sizes).astype(np.float32)
    for key in ("name_tok_idf_cos", "name_token_set_ratio", "name_char3_idf_cos", "addr_tok_idf_cos"):
        f[f"{key}_gap"] = _gap_to_group_max(f[key], starts, sizes)
    for key in ("name_tok_idf_cos", "addr_tok_idf_cos"):
        f[f"{key}_rank"] = _rank_in_group(f[key], starts, sizes)


def add_embedding_features(f: dict, sorted_q: np.ndarray, emb_score: np.ndarray, emb_rank: np.ndarray, block_score: np.ndarray) -> None:
    """In place; same row order contract as add_context_features."""
    starts = group_starts(sorted_q)
    sizes = np.diff(np.r_[starts, len(sorted_q)])
    f["emb_cos"] = emb_score.astype(np.float32)
    f["emb_rank"] = emb_rank.astype(np.float32)
    f["emb_cos_gap"] = _gap_to_group_max(f["emb_cos"], starts, sizes)
    f["emb_cos_rank"] = _rank_in_group(f["emb_cos"], starts, sizes)
    f["from_token"] = (block_score > 0).astype(np.float32)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def chunk_bounds(starts: np.ndarray, n_pairs: int, target: int) -> list:
    """[start, end) slices of ~`target` pairs that never split an S1 group."""
    bounds, start = [], 0
    while start < n_pairs:
        k = np.searchsorted(starts, start + target, side="left")
        end = int(starts[k]) if k < len(starts) else n_pairs
        bounds.append((start, end))
        start = end
    return bounds


def iter_feature_chunks(
    corpus: FeatureCorpus,
    candidates: pd.DataFrame,
    chunk_pairs: int = config.FEATURE_CHUNK_PAIRS,
    progress: Optional[ProgressBar] = None,
) -> Iterator[pd.DataFrame]:
    """Yield one feature DataFrame per S1-aligned chunk of `candidates`
    (columns source1_entity_id, candidate_entity_id, score; plus emb_score and
    emb_rank for a candidate_generation.py union, which adds
    EMBEDDING_FEATURE_COLUMNS)."""
    id_index = pd.Index(corpus.entity_ids)
    q_all = id_index.get_indexer(candidates["source1_entity_id"].to_numpy())
    t_all = id_index.get_indexer(candidates["candidate_entity_id"].to_numpy())
    assert (q_all >= 0).all() and (t_all >= 0).all(), "candidate id missing from the cleaned corpus"
    score_all = candidates["score"].to_numpy(dtype=np.float32)
    has_emb = "emb_score" in candidates.columns
    columns = FEATURE_COLUMNS + (EMBEDDING_FEATURE_COLUMNS if has_emb else [])

    order = np.lexsort((-score_all, q_all))
    q_all, t_all, score_all = q_all[order], t_all[order], score_all[order]
    if has_emb:
        emb_score_all = candidates["emb_score"].to_numpy(dtype=np.float32)[order]
        emb_rank_all = candidates["emb_rank"].to_numpy(dtype=np.float32)[order]

    for start, end in chunk_bounds(group_starts(q_all), len(q_all), chunk_pairs):
        q, t = q_all[start:end], t_all[start:end]
        f = compute_pair_features(corpus, q, t)
        add_context_features(f, q, score_all[start:end])
        if has_emb:
            add_embedding_features(f, q, emb_score_all[start:end], emb_rank_all[start:end], score_all[start:end])
        out = pd.DataFrame(
            {
                "source1_entity_id": corpus.entity_ids[q],
                "candidate_entity_id": corpus.entity_ids[t],
                "country": corpus.countries[q],
                **{col: f[col].astype(np.float32) for col in columns},
            }
        )
        if progress is not None:
            progress.update(len(out))
        yield out


def _sample_candidates(candidates: pd.DataFrame, n_entities: int, seed: int = 0) -> pd.DataFrame:
    s1_ids = candidates["source1_entity_id"].unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(s1_ids, size=min(n_entities, len(s1_ids)), replace=False)
    return candidates[candidates["source1_entity_id"].isin(keep)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pairwise features over blocking candidates.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--sample-entities", type=int, default=None,
        help="Only featurize candidates of N randomly chosen S1 entities (written to <split>_sample<N>/).",
    )
    parser.add_argument(
        "--candidates", default="token_blocking",
        help="Candidate file stem under data/processed/candidates/<split>/ (e.g. a candidate_generation.py union).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    name = args.split if args.candidates == "token_blocking" else f"{args.split}_{args.candidates}"
    if args.sample_entities is not None:
        name = f"{name}_sample{args.sample_entities}"
    dest = config.FEATURES_DIR / name
    if dest.exists() and not args.force:
        print(f"cache exists at {dest}, skipping (use --force to recompute)")
        return

    t0 = time.time()
    candidates = pd.read_parquet(config.CANDIDATES_DIR / args.split / f"{args.candidates}.parquet")
    if args.sample_entities is not None:
        candidates = _sample_candidates(candidates, args.sample_entities)
    print(f"loading corpus for {args.split} ...")
    corpus = build_feature_corpus(args.split)
    print(f"  corpus ready: {len(corpus.entity_ids):,} rows in {time.time() - t0:.0f}s")

    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    bar = ProgressBar(len(candidates), name)
    t1 = time.time()
    for i, chunk in enumerate(iter_feature_chunks(corpus, candidates, progress=bar)):
        chunk.to_parquet(tmp / f"part-{i:05d}.parquet", index=False)
    bar.close()

    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)
    print(f"  {len(candidates):,} pairs x {len(feature_columns_of(dest))} features in {time.time() - t1:.0f}s")
    print(f"  written to: {dest}")


if __name__ == "__main__":
    main()
