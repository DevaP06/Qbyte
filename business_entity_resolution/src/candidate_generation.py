"""Union of token-blocking and embedding-retrieval candidates (plan.md Task
1+3 bullet 6: "union token-blocking candidates + ANN candidates per S1
entity").

    python candidate_generation.py --split train --emb-k 20
    -> data/processed/candidates/<split>/union_<model>_k<emb-k>.parquet

Columns: source1_entity_id, candidate_entity_id, country,
  score       token-blocking score (0 when only embedding retrieval found it)
  emb_score   embedding cosine -- computed for EVERY pair, token-only ones
              included, so it is a complete feature, not a retrieval artifact
  emb_rank    rank among the S1's embedding neighbours (NaN if not in its top-k)

Pairs are keyed as one int64 (s1_row * n_records + candidate_row) instead of
id strings: ~130M-pair unions stay a sort + searchsorted, not a string merge.
This file is the input to features.py --candidates, and its pairs are exactly
what the classifier later scores (and what candidate_pairs.tsv lists).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from embeddings import embed_split, knn_candidates, model_key, record_table
from train_classifier import Progress, _stage


def _codes(index: pd.Index, ids) -> np.ndarray:
    codes = index.get_indexer(ids)
    assert (codes >= 0).all(), "candidate id not among the split's records"
    return codes.astype(np.int64)


def merge_candidate_keys(
    token_keys: np.ndarray, token_score: np.ndarray, emb_keys: np.ndarray, emb_rank: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union of two int64 pair-key sets -> (sorted keys, token score or 0,
    embedding rank or NaN)."""
    keys = np.unique(np.concatenate([token_keys, emb_keys]))
    score = np.zeros(len(keys), dtype=np.float32)
    score[np.searchsorted(keys, token_keys)] = token_score
    rank = np.full(len(keys), np.nan, dtype=np.float32)
    rank[np.searchsorted(keys, emb_keys)] = emb_rank
    return keys, score, rank


def rowwise_cosine(emb: np.ndarray, q: np.ndarray, t: np.ndarray, batch: int = 2_000_000) -> np.ndarray:
    """Dot product of emb[q[i]] and emb[t[i]] (rows are L2-normalized)."""
    out = np.empty(len(q), dtype=np.float32)
    bar = Progress(len(q), "emb cosine", "pairs")
    for s in range(0, len(q), batch):
        a = emb[q[s : s + batch]].astype(np.float32)
        b = emb[t[s : s + batch]].astype(np.float32)
        out[s : s + batch] = np.einsum("ij,ij->i", a, b)
        bar.update(min(s + batch, len(q)))
    bar.close()
    return out


def build_union(split: str, model: str, emb_k: int) -> pd.DataFrame:
    ids, emb_mm = embed_split(split, model)
    n = len(ids)
    index = pd.Index(ids)
    countries = record_table(split)["country"].to_numpy()

    _stage("loading token-blocking candidates")
    token = pq.read_table(
        config.CANDIDATES_DIR / split / "token_blocking.parquet", columns=["source1_entity_id", "candidate_entity_id", "score"]
    )
    tk = _codes(index, token.column("source1_entity_id").to_numpy()) * n + _codes(
        index, token.column("candidate_entity_id").to_numpy()
    )
    tscore = token.column("score").to_numpy().astype(np.float32)
    del token

    _stage(f"loading embedding candidates (top {emb_k})")
    emb_c = knn_candidates(split, model, k=emb_k)
    emb_c = emb_c[emb_c["emb_rank"] < emb_k]
    ek = _codes(index, emb_c["source1_entity_id"].to_numpy()) * n + _codes(index, emb_c["candidate_entity_id"].to_numpy())
    erank = emb_c["emb_rank"].to_numpy().astype(np.float32)
    del emb_c

    _stage("union")
    keys, score, emb_rank = merge_candidate_keys(tk, tscore, ek, erank)
    q, t = keys // n, keys % n
    print(
        f"  token {len(tk):,} + embedding {len(ek):,} -> union {len(keys):,} pairs "
        f"({len(keys) - len(tk):,} new from embeddings)", flush=True,
    )
    del tk, ek, tscore, erank, keys

    _stage("embedding cosine for every pair")
    emb = np.asarray(emb_mm)  # into RAM: random row gathers from a memmap would thrash the disk
    emb_score = rowwise_cosine(emb, q, t)
    del emb

    return pd.DataFrame(
        {
            "source1_entity_id": ids[q],
            "candidate_entity_id": ids[t],
            "country": countries[q],
            "score": score,
            "emb_score": emb_score,
            "emb_rank": emb_rank,
        }
    )


def union_name(model: str, emb_k: int) -> str:
    return f"union_{model_key(model)}_k{emb_k}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Union token-blocking and embedding candidates.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model", default=str(config.MODELS_DIR / "retriever"))
    parser.add_argument("--emb-k", type=int, default=20, help="Embedding neighbours added per S1 entity.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = config.CANDIDATES_DIR / args.split / f"{union_name(args.model, args.emb_k)}.parquet"
    if dest.exists() and not args.force:
        print(f"cache exists at {dest}, skipping (use --force to recompute)")
        return
    t0 = time.time()
    union = build_union(args.split, args.model, args.emb_k)
    _stage(f"writing {dest}")
    tmp = dest.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(union, preserve_index=False), tmp)
    tmp.replace(dest)
    per_entity = union.groupby("source1_entity_id").size()
    print(f"  {len(union):,} pairs, {per_entity.mean():.1f} per S1 entity (max {per_entity.max()})")
    _stage(f"done in {time.time() - t0:.0f}s -- next: features.py --split {args.split} --candidates {dest.stem}")


if __name__ == "__main__":
    main()
