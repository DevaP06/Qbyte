"""Level-2 stacker: combine the GBDT and cross-encoder scores non-linearly (v2.1).

The v2.0 linear blend (fit on India/US dev_val) keeps trusting the GBDT where
the two disagree. In France that is backwards: France has 6.5x the US rate of
"GBDT yes / CE no" pairs, and on those the CE is right most of the time
(substituted business words, different street/number at the same address),
while on "GBDT no / CE yes" the GBDT usually is. This stacker learns *when* to
trust which, from address/name evidence, with no country input.

Trained on dev_val pairs the CE scored (dev_val is unseen by both base models);
2-fold CV by entity gives an honest dev estimate against the linear blend.

    python stack2.py cv      # CPU: CV F0.5, stacker vs linear blend
    python stack2.py apply   # fit on all dev_val, score test, write output/ submission
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

import config
from cross_encoder import CE_DIR, UNION, _logit
from evaluate import per_entity_f05
from labels import entity_truth, load_ground_truth
from postprocess import select_matches
from threshold_tuning import tune_threshold
from train_classifier import ROLE_VAL, Progress, _stage, split_entities

PAIR_FEATURES = [
    "name_token_set_ratio", "name_char3_idf_cos", "name_acronym_match", "name_nonascii_frac_cand",
    "street_core_ratio", "addr_both_present", "addr_digit_conflict", "addr_digit_match", "addr_token_set_ratio",
    "emb_cos", "comp_emb_margin", "comp_char3_margin", "comp_n_s1", "cand_is_source3",
]
KEY = ["source1_entity_id", "candidate_entity_id"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "max_depth": 6,
          "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20, "seed": 42}
ROUNDS = 300
EMB_DIR = config.DATA_PROCESSED_DIR / "embeddings" / "retriever"
EMB_DIM = 384


def _attach_features(scored: pd.DataFrame, features_dir: Path) -> pd.DataFrame:
    """Join PAIR_FEATURES onto the scored pairs, streaming the features parts."""
    want = np.unique(pd.util.hash_pandas_object(scored[KEY], index=False).to_numpy())
    parts = sorted(features_dir.glob("part-*.parquet"))
    bar = Progress(len(parts), "joining features", "parts")
    out = []
    for i, p in enumerate(parts, start=1):
        t = pq.read_table(p, columns=KEY + PAIR_FEATURES).to_pandas()
        h = pd.util.hash_pandas_object(t[KEY], index=False).to_numpy()
        out.append(t[np.isin(h, want)])
        bar.update(i)
    bar.close()
    return scored.merge(pd.concat(out, ignore_index=True), on=KEY, how="left")


def add_coherence(d: pd.DataFrame, split: str) -> pd.DataFrame:
    """Sibling coherence: an S1's true matches are noisy copies of each other.
    For each pair, the embedding cosine between the candidate and the S1's
    strongest OTHER candidate (by GBDT score), and whether they come from
    different sources. NaN when the S1 has no other scored candidate."""
    ids = np.load(EMB_DIR / f"{split}.ids.npy", allow_pickle=True)
    emb = np.memmap(EMB_DIR / f"{split}.f16", dtype=np.float16, mode="r", shape=(len(ids), EMB_DIM))
    order = d.sort_values(["source1_entity_id", "score"], ascending=[True, False])
    top = order.groupby("source1_entity_id")["candidate_entity_id"].agg(lambda s: list(s.iloc[:2]))
    tops = top.reindex(d["source1_entity_id"]).to_numpy()
    cand = d["candidate_entity_id"].to_numpy()
    partner = np.array([t[0] if t[0] != c else (t[1] if len(t) > 1 else None) for t, c in zip(tops, cand)], dtype=object)
    index = pd.Index(ids)
    rc = index.get_indexer(cand)
    has = pd.notna(partner)
    rp = np.full(len(d), -1)
    rp[has] = index.get_indexer(partner[has])
    cos = np.full(len(d), np.nan, dtype=np.float32)
    ok = np.flatnonzero((rc >= 0) & (rp >= 0))
    bar = Progress(len(ok), "coherence", "pairs")
    for s0 in range(0, len(ok), 1_000_000):
        sl = ok[s0 : s0 + 1_000_000]
        a = np.asarray(emb[np.sort(rc[sl])], dtype=np.float32)[np.argsort(np.argsort(rc[sl]))]
        b = np.asarray(emb[np.sort(rp[sl])], dtype=np.float32)[np.argsort(np.argsort(rp[sl]))]
        cos[sl] = np.einsum("ij,ij->i", a, b)
        bar.update(min(s0 + 1_000_000, len(ok)))
    bar.close()
    other_src = np.where(has, pd.Series(cand).str[:2].to_numpy() != pd.Series(partner).astype(str).str[:2].to_numpy(), np.nan)
    return d.assign(coh_cos=cos, coh_other_source=other_src.astype(np.float32))


def _design(d: pd.DataFrame) -> np.ndarray:
    """Stacker inputs: both scores (logits), their S1-local context, pair evidence;
    plus the second CE's logit and its disagreement with the first when two CE
    rounds are stacked."""
    d = d.copy()
    d["lg"], d["lc"] = _logit(d["score"].to_numpy()), _logit(d["ce"].to_numpy())
    g = d.groupby("source1_entity_id")
    d["ce_rank"] = g["ce"].rank(ascending=False, method="min")
    d["ce_gap"] = g["ce"].transform("max") - d["ce"]
    d["gbdt_rank"] = g["score"].rank(ascending=False, method="min")
    d["n_scored"] = g["ce"].transform("size")
    d["disagree"] = d["lg"] - d["lc"]
    cols = ["lg", "lc", "ce_rank", "ce_gap", "gbdt_rank", "n_scored", "disagree"] + PAIR_FEATURES
    if "ce2" in d.columns:
        d["lc2"] = _logit(d["ce2"].fillna(d["ce"]).to_numpy())
        d["ce_rounds_diff"] = d["lc"] - d["lc2"]
        cols += ["lc2", "ce_rounds_diff"]
    if "coh_cos" in d.columns:
        cols += ["coh_cos", "coh_other_source"]
    if "ce_x" in d.columns:  # partially scored extra CE (e5-base): NaN outside its band
        d["lx"] = _logit(d["ce_x"].to_numpy())
        d["x_vs_ce"] = d["lx"] - d["lc"]
        cols += ["lx", "x_vs_ce"]
    return d[cols].to_numpy(np.float32)


def _tags(spec: str) -> list:
    """'_r2,_r3' -> ['_r2', '_r3']. The LAST tag is the primary CE ('ce'); the
    first (if two) is stacked as 'ce2'. An empty tag means round 1."""
    return spec.split(",")


def _suffix(tags: list, bag: int, coherence: bool = False, extra: str = "", val_run: str = "dev_v11") -> str:
    return ("".join(tags) + (f"_bag{bag}" if bag > 1 else "") + ("_coh" if coherence else "") + extra
            + ("" if val_run == "dev_v11" else f"_{val_run}"))


def _gbdt_scores(kind: str, run: str) -> pd.DataFrame:
    """GBDT score per pair from a train_classifier run: dev_val predictions of a
    dev run, or cached test predictions of a final run."""
    path = config.MODELS_DIR / run / "val_predictions.parquet" if kind == "val" else config.PREDICTIONS_DIR / run / f"test_{UNION}.parquet"
    return pd.read_parquet(path, columns=KEY + ["score"])


def _scores(kind: str, tags: list, extra: str = "", gbdt_run: str = "") -> pd.DataFrame:
    base = pd.read_parquet(CE_DIR / f"{kind}_scores{tags[-1]}.parquet")
    if gbdt_run:  # replace the GBDT score frozen in the CE file by this run's
        new = _gbdt_scores(kind, gbdt_run).rename(columns={"score": "gbdt_new"})
        base = base.merge(new, on=KEY, how="left")
        base["score"] = base["gbdt_new"].fillna(base["score"])
        base = base.drop(columns="gbdt_new")
    if extra:
        x = pd.read_parquet(CE_DIR / f"{kind}_scores{extra}.parquet", columns=KEY + ["ce"]).rename(columns={"ce": "ce_x"})
        base = base.merge(x, on=KEY, how="left")
    if len(tags) > 1:
        other = pd.read_parquet(CE_DIR / f"{kind}_scores{tags[0]}.parquet", columns=KEY + ["ce"]).rename(columns={"ce": "ce2"})
        base = base.merge(other, on=KEY, how="left")
    return base


def _val_frame(val_run: str, tags: list, coherence: bool = False, extra: str = "") -> pd.DataFrame:
    d = _attach_features(_scores("val", tags, extra, val_run), config.FEATURES_DIR / f"train_{UNION}")
    return add_coherence(d, "train") if coherence else d


def _fit(X: np.ndarray, y: np.ndarray, bag: int) -> list:
    """`bag` XGBoost stackers with different seeds (different row/column subsamples)."""
    return [xgb.train({**PARAMS, "seed": PARAMS["seed"] + i}, xgb.DMatrix(X, label=y), ROUNDS) for i in range(bag)]


def _predict(boosters: list, X: np.ndarray) -> np.ndarray:
    dm = xgb.DMatrix(X)
    return np.mean([b.predict(dm) for b in boosters], axis=0)


def _macro(pairs: pd.DataFrame, thr: float, val_ent: pd.DataFrame) -> pd.Series:
    m = select_matches(pairs, thr)
    tp = m[m["label"] == 1].groupby("source1_entity_id").size().reindex(val_ent.index).fillna(0)
    n = m.groupby("source1_entity_id").size().reindex(val_ent.index).fillna(0)
    f = pd.Series(per_entity_f05(tp.to_numpy(float), n.to_numpy(float), val_ent["n_true"].to_numpy(float)), index=val_ent.index)
    s = f.groupby(val_ent["country"]).mean()
    s["ALL"] = f.mean()
    return s


def _oof(d: pd.DataFrame, bag: int) -> np.ndarray:
    """2-fold out-of-fold stacker scores, folds by S1 entity."""
    fold = (pd.util.hash_pandas_object(d["source1_entity_id"], index=False).to_numpy() % 2).astype(int)
    X, y = _design(d), d["label"].to_numpy()
    oof = np.zeros(len(d))
    bar = Progress(2, "cv folds", "folds")
    for k in (0, 1):
        oof[fold == k] = _predict(_fit(X[fold != k], y[fold != k], bag), X[fold == k])
        bar.update(k + 1)
    bar.close()
    return oof


def _tuned(val_run: str, d: pd.DataFrame, oof: np.ndarray):
    """(threshold, per-country macro F0.5) of OOF stacker scores over all dev_val entities."""
    val_all = pd.read_parquet(config.MODELS_DIR / val_run / "val_predictions.parquet")
    truth = entity_truth(load_ground_truth())
    val_mask = split_entities(truth["country"].to_numpy()) == ROLE_VAL
    val_ent = truth[val_mask].set_index("source1_entity_id")
    pairs = val_all.merge(d[KEY].assign(new=oof), on=KEY, how="left")
    pairs["score"] = pairs["new"].fillna(pairs["score"])
    codes = pd.Index(truth["source1_entity_id"]).get_indexer(pairs["source1_entity_id"])
    thr = tune_threshold(codes, pairs["score"].to_numpy(), pairs["label"].to_numpy(), truth["n_true"].to_numpy(), val_mask).threshold
    return thr, _macro(pairs, thr, val_ent)


def cv(val_run: str, tags: list, bag: int, coherence: bool = False, extra: str = "") -> None:
    _stage(f"loading dev_val pairs + features (CE {tags}, bag {bag}, coherence {coherence}, extra {extra or '-'})")
    d = _val_frame(val_run, tags, coherence, extra)
    _stage("2-fold CV by entity")
    thr, res = _tuned(val_run, d, _oof(d, bag))
    res["threshold"] = thr
    print(res.to_string(float_format=lambda x: f"{x:.5f}"), flush=True)
    (CE_DIR / f"stack2_cv{_suffix(tags, bag, coherence, extra, val_run)}.json").write_text(json.dumps({k: float(v) for k, v in res.items()}, indent=2))


def apply(val_run: str, test_run: str, out_dir: str, tags: list, bag: int, threshold_shift: float = 0.0,
          coherence: bool = False, extra: str = "") -> None:
    from export import export_submission
    from io_utils import read_cleaned_table
    from predict import summarize

    _stage(f"fitting the stacker on all dev_val pairs (CE {tags}, bag {bag}, coherence {coherence})")
    d = _val_frame(val_run, tags, coherence, extra)
    boosters = _fit(_design(d), d["label"].to_numpy(), bag)
    # threshold from out-of-fold scores: in-sample scores would overstate confidence
    thr, _ = _tuned(val_run, d, _oof(d, bag))
    thr = float(np.clip(thr + threshold_shift, 0.01, 0.99))
    print(f"  stacker threshold (from CV, shift {threshold_shift:+.3f}): {thr:.3f}", flush=True)

    _stage("scoring test pairs")
    tce = _attach_features(_scores("test", tags, extra, test_run), config.FEATURES_DIR / f"test_{UNION}")
    if coherence:
        tce = add_coherence(tce, "test")
    tce["new"] = _predict(boosters, _design(tce))
    test_all = pd.read_parquet(config.PREDICTIONS_DIR / test_run / f"test_{UNION}.parquet")
    test_all = test_all.merge(tce[KEY + ["new"]], on=KEY, how="left")
    test_all["score"] = test_all["new"].fillna(test_all["score"]).astype(np.float32)
    test_all = test_all.drop(columns="new")

    _stage(f"writing submission to {out_dir}")
    matches = select_matches(test_all, thr)
    source1 = read_cleaned_table("test", "source1", columns=["entity_id", "country"]).to_pandas()
    export_submission(Path(out_dir), source1["entity_id"].to_numpy(), test_all, matches)
    print(summarize(source1, test_all, matches).to_string(float_format=lambda x: f"{x:.4f}"))
    sfx = _suffix(tags, bag, coherence, extra, val_run)
    test_all.to_parquet(CE_DIR / f"test_stacked{sfx}.parquet", index=False)  # final pair scores (self-training, threshold probes)
    for i, b in enumerate(boosters):
        b.save_model(str(CE_DIR / f"stack2_model{sfx}_{i}.json"))
    (CE_DIR / f"stack2_config{sfx}.json").write_text(json.dumps(
        {"threshold": thr, "threshold_shift": threshold_shift, "ce_tags": tags, "bag": bag, "coherence": coherence, "extra_tag": extra, "val_run": val_run, "test_run": test_run, "rounds": ROUNDS,
         "params": PARAMS, "features": PAIR_FEATURES}, indent=2))
    _stage("done -- validate, then upload")


def main() -> None:
    parser = argparse.ArgumentParser(description="Level-2 stacker over GBDT + cross-encoder scores.")
    parser.add_argument("cmd", choices=["cv", "apply"])
    parser.add_argument("--val-run", default="dev_v11")
    parser.add_argument("--test-run", default="final_v11")
    parser.add_argument("--out-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--tag", default="", help="CE score set(s): '' = round 1, '_r2', or '_r2,_r3' to stack two rounds.")
    parser.add_argument("--bag", type=int, default=1, help="Seed-bagged stackers averaged.")
    parser.add_argument("--threshold-shift", type=float, default=0.0, help="apply: added to the CV threshold (global, all countries).")
    parser.add_argument("--extra-tag", default="", help="Partially scored extra CE (e.g. '_base'), NaN where unscored.")
    parser.add_argument("--coherence", action="store_true", help="Add sibling-coherence features (embedding cosine to the S1's top other candidate).")
    args = parser.parse_args()
    if args.cmd == "cv":
        cv(args.val_run, _tags(args.tag), args.bag, args.coherence, args.extra_tag)
    else:
        apply(args.val_run, args.test_run, args.out_dir, _tags(args.tag), args.bag, args.threshold_shift, args.coherence, args.extra_tag)


if __name__ == "__main__":
    main()
