"""Per-entity expected-F0.5 decisions instead of one global threshold.

Macro F0.5 is averaged per S1 entity: with k predicted and n_true true matches,
F = 1.25 TP / (k + 0.25 n_true), and empty/empty = 1. The best cut therefore
depends on the entity: a lone candidate at probability p is worth p if
predicted and 1 - p if not (cut 0.5), while a fourth candidate next to three
near-certain matches needs p > ~0.77. A global threshold (0.70) is a
compromise between these. Treating the stacker's pair probabilities as
independent Bernoullis, this keeps for each entity the k top candidates that
maximize its exact expected F0.5 (Jansche 2007; Ye et al. 2012), then gives
each S2/S3 id to its highest-scoring S1 as before.

    python expected_f.py dev                                  # dev_val OOF: global threshold vs expected-F
    python expected_f.py test --out-dir ../../output_ef       # from the saved v2.5 test scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import stack2
from cross_encoder import CE_DIR
from labels import entity_truth, load_ground_truth
from train_classifier import ROLE_VAL, Progress, _stage, split_entities

LO, HI = 0.2, 0.95  # entities with no candidate in [LO, HI] are decided identically either way
MAX_N = 30  # candidates per entity used in the expectation (sorted by p; the rest are ~0)
V25 = "_r2_r3_bag5_coh_base"


def _pb(p: np.ndarray) -> np.ndarray:
    """Poisson-binomial distribution of the number of successes."""
    d = np.array([1.0])
    for x in p:
        d = np.append(d * (1 - x), 0.0) + np.insert(d * x, 0, 0.0)
    return d


def best_k(q: np.ndarray) -> int:
    """q sorted descending: the k maximizing expected F0.5 when predicting the top k."""
    q = q[:MAX_N]
    best, arg = np.prod(1 - q), 0  # k = 0 scores 1 only when there is no true match
    for k in range(1, len(q) + 1):
        a = np.arange(k + 1)[:, None]
        b = np.arange(len(q) - k + 1)[None, :]
        e = (_pb(q[:k])[:, None] * _pb(q[k:])[None, :] * 1.25 * a / (k + 0.25 * (a + b))).sum()
        if e > best:
            best, arg = e, k
    return arg


def select(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Matches from pair scores: expected-F per ambiguous entity, `threshold` elsewhere
    (identical there), then one S1 per S2/S3 id, as postprocess.select_matches."""
    p = pairs[pairs["score"] >= 0.01]
    amb = p.loc[(p["score"] >= LO) & (p["score"] <= HI), "source1_entity_id"].unique()
    is_amb = p["source1_entity_id"].isin(amb)
    keep = [p[~is_amb & (p["score"] >= threshold)]]
    a = p[is_amb].sort_values(["source1_entity_id", "score"], ascending=[True, False], kind="mergesort")
    ids, scores = a["source1_entity_id"].to_numpy(), a["score"].to_numpy(np.float64)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    take = np.zeros(len(a), bool)
    bar = Progress(len(starts), "expected-F decisions", "entities")
    for i, (s, e) in enumerate(zip(starts, ends)):
        take[s : s + best_k(scores[s:e])] = True
        if i % 20_000 == 0:
            bar.update(i)
    bar.close()
    keep.append(a[take])
    m = pd.concat(keep, ignore_index=True)
    return stack2.select_matches(m, -1.0)  # threshold already applied; keeps the dedupe


def dev() -> None:
    oof_path = CE_DIR / f"val_oof{V25}.parquet"
    if oof_path.exists():
        d = pd.read_parquet(oof_path)
    else:
        _stage("v2.5 stacker out-of-fold scores on dev_val (saved for re-runs)")
        d = stack2._val_frame("dev_v11", ["_r2", "_r3"], True, "_base")
        d = d[stack2.KEY].assign(oof=stack2._oof(d, 5))
        d.to_parquet(oof_path, index=False)
    val_all = pd.read_parquet(config.MODELS_DIR / "dev_v11" / "val_predictions.parquet")
    pairs = val_all.merge(d, on=stack2.KEY, how="left")
    pairs["score"] = pairs["oof"].fillna(pairs["score"])
    truth = entity_truth(load_ground_truth())
    val_ent = truth[split_entities(truth["country"].to_numpy()) == ROLE_VAL].set_index("source1_entity_id")
    thr = 0.70  # v2.5's OOF-tuned stacker threshold
    base = stack2._macro(pairs, thr, val_ent)
    _stage("expected-F decisions on dev_val")
    ef = stack2._macro(select(pairs, thr), -1.0, val_ent)
    print(pd.DataFrame({"threshold 0.70": base, "expected-F": ef, "gain": ef - base}).to_string(float_format=lambda x: f"{x:+.5f}" if abs(x) < 0.5 else f"{x:.5f}"))


def test(out_dir: str, stacked: str) -> None:
    from export import write_id_list_tsv
    from io_utils import read_cleaned_table

    t = pd.read_parquet(CE_DIR / f"test_stacked{stacked}.parquet", columns=stack2.KEY + ["score"], filters=[("score", ">=", 0.01)])
    m = select(t, 0.70)
    s1 = read_cleaned_table("test", "source1", columns=["entity_id"]).to_pandas()["entity_id"].to_numpy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_id_list_tsv(out / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), s1, m, "candidate_entity_id")
    print(f"{len(m):,} matches -> {out / 'matching_results.tsv'} (candidate_pairs.tsv is unchanged: use output/'s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-entity expected-F0.5 decisions.")
    parser.add_argument("cmd", choices=["dev", "test"])
    parser.add_argument("--out-dir", default=str(config.REPO_ROOT / "output_ef"))
    parser.add_argument("--stacked", default=V25, help="test_stacked{suffix}.parquet to decide from (e.g. add _iw).")
    args = parser.parse_args()
    dev() if args.cmd == "dev" else test(args.out_dir, args.stacked)


if __name__ == "__main__":
    main()
