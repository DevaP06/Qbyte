"""Candidate-side competition features (cycle 1), added in place to a features dataset.

    python competition_features.py --features train_union_retriever_k20

For every pair, rank this S1 among ALL S1 entities that have the same
candidate, by name trigram cosine, name token-set ratio and embedding cosine,
and record the margin to the best rival. Why: a name-only record is a
candidate of ~37 S1s, and whether it belongs to this one depends on the
rivals, which no pair-local feature can see. On v1.0.0 dev_val name-only pairs,
"best name match among rivals by > 0.05" alone gives 93% precision / 52%
recall vs the model's 87% / 35% (STATUS.md).

Runs over the whole dataset at once (the groups span feature parts), then
rewrites each part with the COMPETITION_FEATURE_COLUMNS appended (replacing
them if present, so re-running is safe). Train and test each compete only
within their own split, so it is the same computation for both.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from features import COMPETITION_FEATURE_COLUMNS
from train_classifier import Progress, _stage

# (source feature, output rank column, output margin column)
RANKED = [
    ("name_char3_idf_cos", "comp_char3_rank", "comp_char3_margin"),
    ("name_token_set_ratio", "comp_name_tsr_rank", "comp_name_tsr_margin"),
    ("emb_cos", "comp_emb_rank", "comp_emb_margin"),
]


def competition(codes: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: (rank among rows sharing its code, 0 = highest value;
    value minus the best OTHER row's value -- >0 wins, 0 ties, <0 loses,
    NaN when the code has no other row; group size). NaN values rank last."""
    v = np.where(np.isnan(values), -np.inf, values).astype(np.float64)
    order = np.lexsort((-v, codes))
    cs, vs = codes[order], v[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    sizes = np.diff(np.r_[starts, len(cs)])
    pos = np.arange(len(cs)) - np.repeat(starts, sizes)
    top1 = np.repeat(vs[starts], sizes)
    second_idx = np.minimum(starts + 1, len(cs) - 1)
    top2 = np.repeat(np.where(sizes > 1, vs[second_idx], np.nan), sizes)
    with np.errstate(invalid="ignore"):  # -inf - -inf when NaN inputs meet; set to NaN below
        margin_sorted = np.where(pos == 0, vs - top2, vs - top1)
    margin_sorted[~np.isfinite(margin_sorted)] = np.nan  # no rival, or a -inf (NaN input) involved
    rank, margin, size = np.empty(len(cs), np.float32), np.empty(len(cs), np.float32), np.empty(len(cs), np.float32)
    rank[order], margin[order], size[order] = pos, margin_sorted, np.repeat(sizes, sizes)
    return rank, margin, size


def add_competition_features(features_dir: Path) -> None:
    parts = sorted(features_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no feature parts in {features_dir}")
    present = set(pq.read_schema(parts[0]).names)
    ranked = [r for r in RANKED if r[0] in present]

    _stage(f"loading candidate ids + {len(ranked)} similarity columns from {len(parts)} parts")
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    bar = Progress(total, "loading", "pairs")
    cand_chunks, value_chunks, lengths, done = [], {r[0]: [] for r in ranked}, [], 0
    for p in parts:
        t = pq.read_table(p, columns=["candidate_entity_id", *[r[0] for r in ranked]])
        cand_chunks.append(t.column("candidate_entity_id").combine_chunks())
        for src, *_ in ranked:
            value_chunks[src].append(t.column(src).to_numpy())
        lengths.append(t.num_rows)
        done += t.num_rows
        bar.update(done, f"{len(lengths)}/{len(parts)} parts")
    bar.close()

    _stage("computing competition ranks and margins")
    codes = pc.dictionary_encode(pa.concat_arrays(cand_chunks)).indices.to_numpy()
    del cand_chunks
    out = {}
    bar = Progress(len(ranked), "competition", "features")
    for i, (src, rank_col, margin_col) in enumerate(ranked, start=1):
        rank, margin, size = competition(codes, np.concatenate(value_chunks.pop(src)))
        out[rank_col], out[margin_col] = rank, margin
        out["comp_n_s1"] = size
        bar.update(i, src)
    bar.close()
    del codes

    _stage("writing columns back into each part")
    names = [c for c in COMPETITION_FEATURE_COLUMNS if c in out]
    bar = Progress(len(parts), "writing", "parts")
    offset = 0
    for i, (p, n) in enumerate(zip(parts, lengths), start=1):
        t = pq.read_table(p)
        t = t.drop_columns([c for c in names if c in t.column_names])
        for c in names:
            t = t.append_column(c, pa.array(out[c][offset : offset + n]))
        tmp = p.with_suffix(".parquet.tmp")
        pq.write_table(t, tmp)
        tmp.replace(p)
        offset += n
        bar.update(i)
    bar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add candidate-side competition features to a features dataset.")
    parser.add_argument("--features", required=True, help="Directory name under data/processed/features/.")
    args = parser.parse_args()
    t0 = time.time()
    add_competition_features(config.FEATURES_DIR / args.features)
    _stage(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
