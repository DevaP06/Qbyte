"""Country-partitioned inverted-index token blocking (plan.md build-order step 2).

Input: the cleaned Parquet output of clean_all.py (data/processed/<split>/
source{1,2,3}.parquet) -- never re-normalizes S1/S2/S3 text, never reads the
raw TSVs. Output: for each S1 entity, up to `BLOCKING_MAX_CANDIDATES_PER_ENTITY`
S2/S3 candidate ids, ranked by a coarse IDF-weighted token-overlap score,
written to data/processed/candidates/<split>/token_blocking.parquet -- an
*intermediate* cache, not `output/candidate_pairs.tsv`. Per PROBLEM_STATEMENT.md,
`candidate_pairs.tsv` must be "the exact final candidate set the classifier ran
over," and plan.md's own module split keeps blocking (this file) separate from
`candidate_generation.py`, which will union this with tomorrow's ANN candidates
before writing that file. Writing to output/ now would ship a set that later
gets superseded.

--------------------------------------------------------------------------
SELF-CRITIQUE: measured deviations from the literal plan.md text
--------------------------------------------------------------------------

**Deviation 1 -- name tokens alone are not enough.** plan.md's Task 1+3
bullet 3 says: "Primary blocking = inverted-index token blocking on
normalized name tokens (with the dominant generic/legal tokens ... excluded
as blocking keys)." Implemented literally and measured against the full
`train_ground_truth.tsv` (7,638,365 pairs), NAME-TOKEN-ONLY blocking has a
hard recall ceiling (the fraction of GT pairs sharing at least one surviving
key -- the best any ranking/cap could ever achieve) of:

    US: 91.4%      India: 74.9%

India falls well short of the plan's own go/no-go bar ("target comfortably
>85-90% before moving on"). Root cause, inspected directly: excluding only
the ~16 named legal/generic tokens (pvt/ltd/services/india/...) is not
enough -- Indian business names lean heavily on common-but-not-pre-named
business words (co, industries, enterprises, ventures, trading, exports),
and (measured earlier in this project) S1 India names are 100% Latin script
while matching S2/S3 records run 13-24% Devanagari/Kannada -- there is often
no shared Latin name token at all, so name-token overlap cannot succeed on
that subset no matter how the stopword list is tuned.

FIX: this module also indexes address tokens (from the existing
`address_normalized` field, tokenized with normalize.py's own tokenizer --
no second tokenizer) as a second blocking-key source, unioned into the same
country-partitioned inverted index. Still literally inverted-index
token-overlap blocking -- no ANN, no embeddings, no character n-grams/MinHash
-- just a second existing field. Measured recall ceiling with name+address
combined: US 99.99%, India 99.23% (on a 3,000-entity sample; see below for
the full-corpus numbers actually shipped). This also closes almost exactly
the segment name-only blocking cannot reach: recall@40 on non-Latin-script
India candidates measured at 79.8% via address tokens vs 0.05% via name
tokens alone.

**Deviation 2 -- a per-key document-frequency cap needs no per-row
"fallback," and needs to be much smaller than a percentage of the corpus.**
plan.md bullet 6 gives cap-per-entity "e.g. ~80" as a safety valve for
pathological shards, implicitly assuming the named-stopword exclusion alone
keeps a token-blocking candidate list small. Measured directly: it does not
-- even after excluding the named generic/legal tokens, common-but-unnamed
words (co, s, partners, and, trading, road, street, city/state names on the
address side) still occur in tens to hundreds of thousands of records, so an
un-capped union of postings would produce on the order of tens of thousands
of candidates per entity before any ranking even starts.

Two designs were tried and measured before settling on the one shipped here:

  1. *Percentage-of-corpus df cap with a per-row fallback to that row's full
     uncapped token set whenever capping emptied it.* This looked reasonable
     at a 3,000-row sample, but at full country-partition scale a handful of
     pathological rows (addresses built entirely from generic geography
     words with literally no rare token) triggered the fallback and each
     ballooned to 400,000+ candidates on their own, which would have made a
     full run computationally infeasible (a `numpy` allocation for a
     3.5-billion-entry sparse-matrix add hit the machine's 16.8GB RAM limit
     during a full-partition test on the US shard, at settings that looked
     completely safe on a sample). Rejected.

  2. *Two-layer absolute cap, no fallback* (what ships): layer 1 drops a
     token as a blocking key everywhere in the partition once its document
     frequency exceeds `BLOCKING_MAX_TOKEN_DOCUMENT_FREQUENCY` (an absolute
     count, not a percentage -- so it needs no knowledge of a partition's
     size, and applies identically to a country never seen before, e.g.
     France in test); layer 2 keeps only each record's
     `BLOCKING_KEYS_PER_SIDE` rarest-AND-PRESENT surviving tokens per side
     (name, address) -- "rarest" is by document frequency among tokens that
     actually occur at least once on the target side; a token absent from
     the target entirely (df=0) is excluded from this ranking rather than
     treated as maximally rare, since selecting it would waste a slot on a
     key guaranteed to produce zero overlap. If capping empties a record's
     key set entirely, it gets zero token-blocking candidates -- a small,
     bounded, measured cost, not an unbounded one: an actual full-corpus run
     (config.BLOCKING_MAX_TOKEN_DOCUMENT_FREQUENCY = 20,000,
     config.BLOCKING_KEYS_PER_SIDE = 10) left only 47 of train's 2,206,821
     S1 entities (0.002%) with zero candidates. Full train results (`python
     blocking_token.py --split train`):

         India: S1=883,188   T=4,133,346   GT pairs=3,059,843
                ceiling=98.39%   recall@40=87.10%   pairs=35,316,995
         US:    S1=1,323,633  T=6,186,873   GT pairs=4,578,522
                ceiling=99.17%   recall@40=94.69%   pairs=52,942,491

     Both clear the plan's own bar; India sits at the lower end of
     "comfortably" rather than deep past it -- expected, since this is
     candidate GENERATION only, and the exact remaining gap (non-Latin-script
     India names) is what plan.md's own next blocking layer, dense
     multilingual retrieval, is designed to recover tomorrow. This module
     does not claim to close that gap on its own. Spot-checked against a
     concrete example inspected earlier in this project (S1-965667, GT
     matches S2-681193310/S2-743505751/S3-775321672/S3-11291185/S3-860443364):
     all 5 true matches are present among its candidates.

     **Third finding, not a deviation but worth surfacing:** total train
     candidate volume is 88,259,486 pairs -- above plan.md's own sizing
     target ("keep total pairs well under ~50-60M"). Every entity in both
     partitions averages almost exactly `BLOCKING_MAX_CANDIDATES_PER_ENTITY`
     (40.0), so total volume here is essentially `40 x 2,206,821 S1 entities`
     regardless of how "good" the ranking is -- the only ways to bring it
     under the target are a smaller K (recall@10 was measured at
     91.22%/81.24% for US/India at similar cap settings, meaningfully below
     the recall@40 numbers above) or accepting the classifier stage's
     feature computation runs on a somewhat larger candidate set than the
     plan estimated. Left at 40 rather than silently cut, since F_0.5's
     precision weighting means the classifier (not blocking) is the layer
     meant to reject false positives, and recall lost at blocking time can
     never be recovered downstream. Flagged here for a decision when
     features.py's actual runtime on this volume is measured, rather than
     guessed at now.

plan.md's cap-per-entity "e.g. ~80" is also reused as a hint, not a rule: at
this corpus's actual scale, capping at 80 per entity across all of train
would produce roughly 176M candidate pairs -- past the plan's own sizing
target ("keep total pairs well under ~50-60M"). `BLOCKING_MAX_CANDIDATES_PER_ENTITY`
defaults to 40 instead, chosen from the measured recall/volume tradeoff
(recall@40 above vs. recall@80, which gains only 1-3 points at roughly double
the pair count).

Every constant above is a plain number in config.py, not a country-keyed
table -- the same code path and the same two thresholds run for US, India,
and France alike.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

import config
from clean_all import ProgressBar
from io_utils import read_cleaned_table
from normalize import GENERIC_BLOCKING_STOPWORDS, _tokenize


# --------------------------------------------------------------------------
# Token incidence matrices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenIndex:
    """Binary (record x blocking-key) incidence matrix plus its vocabulary."""

    matrix: sp.csr_matrix
    key_to_col: dict


def _flatten_token_list_column(list_array: pa.Array) -> tuple[pa.Array, np.ndarray]:
    """Flatten a pyarrow list<string> column to (flat tokens, owning row id)."""
    lens = pc.list_value_length(list_array).to_numpy(zero_copy_only=False).astype(np.int64)
    row_ids = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
    return pc.list_flatten(list_array), row_ids


def _tokenize_address_column(address_normalized: np.ndarray) -> tuple[pa.Array, np.ndarray]:
    """Tokenize a plain-string address column with normalize.py's own
    separator-based tokenizer -- reused, not reimplemented."""
    token_lists = [_tokenize(s) if s else () for s in address_normalized]
    lens = np.fromiter((len(t) for t in token_lists), dtype=np.int64, count=len(token_lists))
    flat = pa.array([tok for toks in token_lists for tok in toks], type=pa.string())
    row_ids = np.repeat(np.arange(len(token_lists), dtype=np.int64), lens)
    return flat, row_ids


def build_token_index(
    flat_tokens: pa.Array, row_ids: np.ndarray, n_rows: int, exclude: frozenset = frozenset()
) -> TokenIndex:
    """Binary (n_rows x n_keys) incidence matrix from a flat token stream.

    A token repeated within one row still counts once -- blocking only cares
    whether a key is present. `exclude` is checked once per distinct token
    (via dictionary encoding), not once per occurrence.
    """
    if len(flat_tokens) == 0:
        return TokenIndex(sp.csr_matrix((n_rows, 0), dtype=np.float32), {})

    encoded = pc.dictionary_encode(flat_tokens)
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    vocab = encoded.dictionary.to_pylist()

    key_to_col: dict = {}
    col_of_code = np.full(len(vocab), -1, dtype=np.int64)
    for i, tok in enumerate(vocab):
        if tok not in exclude:
            col_of_code[i] = key_to_col.setdefault(tok, len(key_to_col))

    cols = col_of_code[codes]
    keep = cols >= 0
    data = np.ones(int(keep.sum()), dtype=np.float32)
    matrix = sp.csr_matrix((data, (row_ids[keep], cols[keep])), shape=(n_rows, len(key_to_col)))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return TokenIndex(matrix, key_to_col)


# --------------------------------------------------------------------------
# Two-layer key selection (see module docstring for why)
# --------------------------------------------------------------------------


def _document_frequency(matrix: sp.csr_matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=0)).ravel()


def _idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    idf = np.zeros(len(df), dtype=np.float32)
    present = df > 0
    idf[present] = np.log(n_docs / df[present]).astype(np.float32)
    return idf


def select_blocking_keys(
    query: sp.csr_matrix, target_df: np.ndarray, max_document_frequency: int, keys_per_row: int
) -> sp.csr_matrix:
    """Layer 1: drop any column with target_df == 0 (absent from this
    country's target side -- guaranteed zero overlap, so "rarest" layer 2
    below must never prefer it over a token that actually occurs) or
    target_df > max_document_frequency (too common; no fallback -- see
    module docstring). Layer 2: keep each row's `keys_per_row` lowest-df
    survivors. A row can end up with zero keys; that is an accepted,
    measured, small cost, not a bug.
    """
    keep_col = (target_df >= 1) & (target_df <= max_document_frequency)
    indptr = query.indptr
    rows_out, cols_out = [], []
    for r in range(query.shape[0]):
        start, end = indptr[r], indptr[r + 1]
        if start == end:
            continue
        cols = query.indices[start:end]
        cols = cols[keep_col[cols]]
        if len(cols) == 0:
            continue
        if len(cols) > keys_per_row:
            d = target_df[cols]
            top = np.argpartition(d, keys_per_row - 1)[:keys_per_row]
            cols = cols[top]
        rows_out.append(np.full(len(cols), r, dtype=np.int64))
        cols_out.append(cols)
    if not rows_out:
        return sp.csr_matrix(query.shape, dtype=np.float32)
    r_ = np.concatenate(rows_out)
    c_ = np.concatenate(cols_out)
    return sp.csr_matrix((np.ones(len(r_), dtype=np.float32), (r_, c_)), shape=query.shape)


def _topk_per_row(scored: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """At most `k` highest-scoring columns per row; ties broken by column
    index for determinism. Returns parallel (row, col, score) arrays."""
    indptr = scored.indptr
    row_out, col_out, score_out = [], [], []
    for r in range(scored.shape[0]):
        start, end = indptr[r], indptr[r + 1]
        if start == end:
            continue
        cols = scored.indices[start:end]
        vals = scored.data[start:end]
        if end - start > k:
            top = np.argpartition(-vals, k - 1)[:k]
            cols, vals = cols[top], vals[top]
        order = np.lexsort((cols, -vals))
        row_out.append(np.full(len(order), r, dtype=np.int64))
        col_out.append(cols[order])
        score_out.append(vals[order])
    if not row_out:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    return np.concatenate(row_out), np.concatenate(col_out), np.concatenate(score_out)


# --------------------------------------------------------------------------
# Per-country scoring, batched + thread-parallel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SidePartition:
    """One blocking-key source (name or address) restricted to one country."""

    selected_query: sp.csr_matrix  # (n_query x n_keys), post two-layer selection
    idf_weighted_target_T: sp.csr_matrix  # (n_keys x n_target), pre-scaled by IDF


def prepare_side(
    full_matrix: sp.csr_matrix,
    query_rows: np.ndarray,
    target_rows: np.ndarray,
    max_document_frequency: int,
    keys_per_row: int,
) -> SidePartition:
    query = full_matrix[query_rows]
    target = full_matrix[target_rows]
    df = _document_frequency(target)
    idf = _idf(df, n_docs=len(target_rows))
    selected_query = select_blocking_keys(query, df, max_document_frequency, keys_per_row)
    idf_weighted_target_T = (target @ sp.diags(idf)).T.tocsr()
    return SidePartition(selected_query=selected_query, idf_weighted_target_T=idf_weighted_target_T)


def combine_sides(sides: list[SidePartition]) -> SidePartition:
    """Horizontally stack each side's query keys and vertically stack the
    matching target rows into ONE side, so name+address score in a single
    matmul (sum-over-inner-dimension already sums the two sides' overlap)
    instead of two matmuls plus a separate sparse add.

    This isn't just tidier: an explicit `A + B` on two sparse matrices asks
    scipy to allocate a worst-case nnz(A)+nnz(B) buffer *before* it knows how
    much of that overlaps -- measured directly to be the actual OOM cause at
    full scale (a single 10,000-row batch needed a ~265M-entry, ~1GB buffer
    for that one add). A single matmul's output is sized by its own true
    result, not a summed worst case.
    """
    query = sp.hstack([s.selected_query for s in sides]).tocsr()
    target_T = sp.vstack([s.idf_weighted_target_T for s in sides]).tocsr()
    return SidePartition(selected_query=query, idf_weighted_target_T=target_T)


def score_batch(side: SidePartition, start: int, end: int) -> sp.csr_matrix:
    return (side.selected_query[start:end] @ side.idf_weighted_target_T).tocsr()


def block_country_partition(
    side: SidePartition,
    n_query: int,
    k: int = config.BLOCKING_MAX_CANDIDATES_PER_ENTITY,
    batch_size: int = config.BLOCKING_QUERY_BATCH_SIZE,
    num_threads: int = config.BLOCKING_NUM_THREADS,
    progress_label: str = "blocking",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score = IDF-weighted overlap on the selected keys (name+address
    already combined into one `side` via `combine_sides`); top-k per query
    row. Batched so peak memory is bounded by (batch_size x
    candidates_per_row), never by the full partition; batches are scored on
    a thread pool since scipy's sparse matmul releases the GIL.

    Each worker reduces its batch's scored matrix to its (tiny) top-k result
    before returning -- not just "score in batches". A first version scored
    a batch and returned the full (batch_size x n_target) matrix from the
    worker, then ran `_topk_per_row` afterward in the main thread; that
    still bounded any *one* batch's memory, but `list(pool.map(...))`
    collects every worker's return value before any of them can be
    discarded, so with ~1,300+ batches at ~226MB each it tried to hold the
    entire partition's worth of scored matrices at once (~300GB) and OOM'd.
    Reducing to top-k inside the worker means what accumulates across
    batches is a handful of ints/floats per row, not a whole matrix.
    """
    bounds = list(range(0, n_query, batch_size))
    batches = [(s, min(s + batch_size, n_query)) for s in bounds]

    def score_and_reduce(bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, end = bounds
        scored = score_batch(side, start, end)
        rows, cols, scores = _topk_per_row(scored, k)
        return rows + start, cols, scores

    bar = ProgressBar(n_query, progress_label)
    results = []
    if num_threads > 1 and len(batches) > 1:
        with ThreadPoolExecutor(num_threads) as pool:
            for (start, end), res in zip(batches, pool.map(score_and_reduce, batches)):
                results.append(res)
                bar.update(end - start)
    else:
        for b in batches:
            results.append(score_and_reduce(b))
            bar.update(b[1] - b[0])
    bar.close()

    if not results:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    all_rows = np.concatenate([r[0] for r in results])
    all_cols = np.concatenate([r[1] for r in results])
    all_scores = np.concatenate([r[2] for r in results])
    return all_rows, all_cols, all_scores


# --------------------------------------------------------------------------
# Driver: load cleaned data, build indices, block every country partition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusIndex:
    entity_ids: np.ndarray  # length n1 + n2 + n3, source1 first
    countries: np.ndarray
    n_source1: int
    name_index: TokenIndex
    address_index: TokenIndex


def build_corpus_index(split: str) -> CorpusIndex:
    """Build name+address token indices spanning source1/2/3 of one split.

    Row layout: rows [0, n1) are source1, followed by source2, then source3
    -- this row order is the only thing that ties the two index matrices,
    `entity_ids`, and `countries` together.
    """
    sources = ("source1", "source2", "source3")
    entity_id_parts, country_parts = [], []
    name_flat_parts, name_row_parts = [], []
    addr_text_parts = []
    offset = 0
    n_source1 = 0
    for i, source in enumerate(sources):
        table = read_cleaned_table(
            split, source, columns=["entity_id", "country", "name_tokens", "address_normalized"]
        )
        n_rows = table.num_rows
        if i == 0:
            n_source1 = n_rows
        entity_id_parts.append(table.column("entity_id").to_numpy(zero_copy_only=False))
        country_parts.append(table.column("country").to_numpy(zero_copy_only=False))

        flat, rows = _flatten_token_list_column(table.column("name_tokens").combine_chunks())
        name_flat_parts.append(flat)
        name_row_parts.append(rows + offset)

        addr_text_parts.append(table.column("address_normalized").to_numpy(zero_copy_only=False))
        offset += n_rows

    total_rows = offset
    entity_ids = np.concatenate(entity_id_parts)
    countries = np.concatenate(country_parts)

    name_flat = pa.concat_arrays(name_flat_parts)
    name_rows = np.concatenate(name_row_parts)
    name_index = build_token_index(name_flat, name_rows, total_rows, exclude=GENERIC_BLOCKING_STOPWORDS)

    addr_flat_parts, addr_row_parts = [], []
    row_offset = 0
    for texts in addr_text_parts:
        flat, rows = _tokenize_address_column(texts)
        addr_flat_parts.append(flat)
        addr_row_parts.append(rows + row_offset)
        row_offset += len(texts)
    addr_flat = pa.concat_arrays(addr_flat_parts)
    addr_rows = np.concatenate(addr_row_parts)
    address_index = build_token_index(addr_flat, addr_rows, total_rows, exclude=frozenset())

    return CorpusIndex(
        entity_ids=entity_ids,
        countries=countries,
        n_source1=n_source1,
        name_index=name_index,
        address_index=address_index,
    )


@dataclass
class PartitionDiagnostics:
    country: str
    n_query: int
    n_target: int
    n_candidate_pairs: int
    avg_candidates_per_entity: float
    reduction_ratio: float
    recall_ceiling: Optional[float] = None
    recall_at_k: Optional[float] = None
    n_gt_pairs: Optional[int] = None


def run_blocking(
    split: str,
    max_document_frequency: int = config.BLOCKING_MAX_TOKEN_DOCUMENT_FREQUENCY,
    keys_per_side: int = config.BLOCKING_KEYS_PER_SIDE,
    max_candidates_per_entity: int = config.BLOCKING_MAX_CANDIDATES_PER_ENTITY,
    ground_truth: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, list[PartitionDiagnostics]]:
    """Run country-partitioned token blocking for one split (train or test).

    Country partitioning needs no branch: every distinct value in the
    `country` column gets its own independent index/query pass, whatever
    that value is -- this is what makes a country absent from train (France,
    only in test) fall out of the same loop with zero special-casing.

    If `ground_truth` is given (source1_entity_id, matched_id) pairs, the
    returned diagnostics include recall ceiling / recall@k per country --
    this is the plan's step-2 go/no-go measurement, computed here rather
    than in a separate script so it always reflects the exact index/keys
    actually used to produce the candidates.
    """
    corpus = build_corpus_index(split)
    n1 = corpus.n_source1
    full_id_index: Optional[pd.Index] = None
    if ground_truth is not None:
        full_id_index = pd.Index(corpus.entity_ids)

    result_rows, result_cols, result_scores, result_countries = [], [], [], []
    diagnostics: list[PartitionDiagnostics] = []

    for country in sorted(set(corpus.countries[:n1].tolist())):
        query_rows = np.flatnonzero(corpus.countries[:n1] == country)
        target_rows_local = np.flatnonzero(corpus.countries[n1:] == country)
        target_rows = target_rows_local + n1

        name_side = prepare_side(
            corpus.name_index.matrix, query_rows, target_rows, max_document_frequency, keys_per_side,
        )
        addr_side = prepare_side(
            corpus.address_index.matrix, query_rows, target_rows, max_document_frequency, keys_per_side,
        )
        combined_side = combine_sides([name_side, addr_side])

        rows, cols, scores = block_country_partition(
            combined_side, n_query=len(query_rows), k=max_candidates_per_entity, progress_label=f"block {country}"
        )

        query_entity_ids = corpus.entity_ids[query_rows][rows]
        target_entity_ids = corpus.entity_ids[target_rows][cols]
        result_rows.append(query_entity_ids)
        result_cols.append(target_entity_ids)
        result_scores.append(scores)
        result_countries.append(np.full(len(rows), country))

        n_query = len(query_rows)
        n_target = len(target_rows)
        n_pairs = len(rows)
        diag = PartitionDiagnostics(
            country=country,
            n_query=n_query,
            n_target=n_target,
            n_candidate_pairs=n_pairs,
            avg_candidates_per_entity=n_pairs / n_query if n_query else 0.0,
            reduction_ratio=1.0 - (n_pairs / n_query / n_target) if n_query and n_target else 0.0,
        )

        if ground_truth is not None:
            gt_country = ground_truth[ground_truth["source1_entity_id"].isin(
                corpus.entity_ids[query_rows]
            )]
            if len(gt_country):
                # Resolve every GT pair to (local query row, global target row)
                # first, using the whole-corpus id index -- cheap (GT-pair-sized,
                # not partition-sized) and works whatever id space the match
                # falls in.
                query_local_index = pd.Index(corpus.entity_ids[query_rows])
                q_local = query_local_index.get_indexer(gt_country["source1_entity_id"].to_numpy())
                t_global = full_id_index.get_indexer(gt_country["matched_id"].to_numpy())
                valid = (q_local >= 0) & (t_global >= n1)  # matched id must resolve to a target-side row
                q_local, t_global = q_local[valid], t_global[valid]

                # Confirm the target row actually falls inside *this* country's
                # target partition -- expected always, per the grounding fact of
                # zero cross-country GT pairs (plan.md), checked rather than
                # assumed.
                t_local = pd.Index(target_rows).get_indexer(t_global)
                in_partition = t_local >= 0
                gt_q, gt_t_local, gt_t_global = q_local[in_partition], t_local[in_partition], t_global[in_partition]

                # Ceiling: does *any* selected key overlap exist for this exact
                # GT pair? Computed only over the GT pairs themselves (a few
                # hundred thousand rows, not the full n_query x n_target space)
                # -- the naive "score the whole partition" approach was tried
                # and OOM'd on the full US shard (see module docstring).
                def _shares_a_key(side: SidePartition, raw_index: sp.csr_matrix) -> np.ndarray:
                    q_rows = side.selected_query[gt_q]
                    t_rows = raw_index[gt_t_global]
                    return np.asarray(q_rows.multiply(t_rows).sum(axis=1)).ravel() > 0

                ceiling_hits = _shares_a_key(name_side, corpus.name_index.matrix) | _shares_a_key(
                    addr_side, corpus.address_index.matrix
                )

                cand_key = pd.Index(rows.astype(np.int64) * n_target + cols.astype(np.int64))
                gt_key = gt_q.astype(np.int64) * n_target + gt_t_local.astype(np.int64)
                recall_hits = cand_key.get_indexer(gt_key) >= 0

                diag.n_gt_pairs = len(gt_country)
                diag.recall_ceiling = float(ceiling_hits.sum()) / len(gt_country)
                diag.recall_at_k = float(recall_hits.sum()) / len(gt_country)

        diagnostics.append(diag)

    candidates = pd.DataFrame(
        {
            "source1_entity_id": np.concatenate(result_rows),
            "candidate_entity_id": np.concatenate(result_cols),
            "score": np.concatenate(result_scores),
            "country": np.concatenate(result_countries),
        }
    )
    return candidates, diagnostics


def _load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(
        config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["ground_truth"], sep=config.TSV_SEP, dtype=str
    )
    gt = gt.dropna(subset=[config.GT_MATCHES_COL])
    gt = gt[gt[config.GT_MATCHES_COL].str.strip() != ""]
    gt = gt.assign(matched_id=gt[config.GT_MATCHES_COL].str.split(","))
    gt = gt.explode("matched_id")[[config.GT_SOURCE1_COL, "matched_id"]]
    return gt.rename(columns={config.GT_SOURCE1_COL: "source1_entity_id"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run country-partitioned token blocking.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = config.CANDIDATES_DIR / args.split / "token_blocking.parquet"
    if dest.exists() and not args.force:
        print(f"cache exists at {dest}, skipping (use --force to recompute)")
        return

    ground_truth = _load_ground_truth() if args.split == "train" else None

    t0 = time.time()
    candidates, diagnostics = run_blocking(args.split, ground_truth=ground_truth)
    elapsed = time.time() - t0

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    candidates.to_parquet(tmp, index=False)
    tmp.replace(dest)

    print(f"\n=== token blocking: {args.split} ({elapsed:.0f}s) ===")
    total_pairs = 0
    for d in diagnostics:
        total_pairs += d.n_candidate_pairs
        line = (
            f"  [{d.country:8s}] S1={d.n_query:,} T={d.n_target:,}  "
            f"pairs={d.n_candidate_pairs:,}  avg/entity={d.avg_candidates_per_entity:.1f}  "
            f"reduction={d.reduction_ratio * 100:.4f}%"
        )
        if d.recall_ceiling is not None:
            line += (
                f"  | GT pairs={d.n_gt_pairs:,}  ceiling={d.recall_ceiling * 100:.2f}%  "
                f"recall@{config.BLOCKING_MAX_CANDIDATES_PER_ENTITY}={d.recall_at_k * 100:.2f}%"
            )
        print(line)
    print(f"  TOTAL candidate pairs: {total_pairs:,}")
    print(f"  written to: {dest}")


if __name__ == "__main__":
    main()
