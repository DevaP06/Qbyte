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


def _design(d: pd.DataFrame) -> np.ndarray:
    """Stacker inputs: both scores (logits), their S1-local context, pair evidence."""
    d = d.copy()
    d["lg"], d["lc"] = _logit(d["score"].to_numpy()), _logit(d["ce"].to_numpy())
    g = d.groupby("source1_entity_id")
    d["ce_rank"] = g["ce"].rank(ascending=False, method="min")
    d["ce_gap"] = g["ce"].transform("max") - d["ce"]
    d["gbdt_rank"] = g["score"].rank(ascending=False, method="min")
    d["n_scored"] = g["ce"].transform("size")
    d["disagree"] = d["lg"] - d["lc"]
    cols = ["lg", "lc", "ce_rank", "ce_gap", "gbdt_rank", "n_scored", "disagree"] + PAIR_FEATURES
    return d[cols].to_numpy(np.float32)


def _val_frame(val_run: str, tag: str = "") -> pd.DataFrame:
    vce = pd.read_parquet(CE_DIR / f"val_scores{tag}.parquet")
    return _attach_features(vce, config.FEATURES_DIR / f"train_{UNION}")


def _macro(pairs: pd.DataFrame, thr: float, val_ent: pd.DataFrame) -> pd.Series:
    m = select_matches(pairs, thr)
    tp = m[m["label"] == 1].groupby("source1_entity_id").size().reindex(val_ent.index).fillna(0)
    n = m.groupby("source1_entity_id").size().reindex(val_ent.index).fillna(0)
    f = pd.Series(per_entity_f05(tp.to_numpy(float), n.to_numpy(float), val_ent["n_true"].to_numpy(float)), index=val_ent.index)
    s = f.groupby(val_ent["country"]).mean()
    s["ALL"] = f.mean()
    return s


def cv(val_run: str, tag: str = "") -> None:
    from sklearn.linear_model import LogisticRegression

    _stage("loading dev_val pairs + features")
    d = _val_frame(val_run, tag)
    val_all = pd.read_parquet(config.MODELS_DIR / val_run / "val_predictions.parquet")
    truth = entity_truth(load_ground_truth())
    val_mask = split_entities(truth["country"].to_numpy()) == ROLE_VAL
    val_ent = truth[val_mask].set_index("source1_entity_id")
    codes_all = pd.Index(truth["source1_entity_id"]).get_indexer(val_all["source1_entity_id"])

    fold = (pd.util.hash_pandas_object(d["source1_entity_id"], index=False).to_numpy() % 2).astype(int)
    X, y = _design(d), d["label"].to_numpy()
    oof_stack, oof_lin = np.zeros(len(d)), np.zeros(len(d))
    _stage("2-fold CV by entity")
    bar = Progress(2, "cv folds", "folds")
    for k in (0, 1):
        tr, te = fold != k, fold == k
        booster = xgb.train(PARAMS, xgb.DMatrix(X[tr], label=y[tr]), ROUNDS)
        oof_stack[te] = booster.predict(xgb.DMatrix(X[te]))
        lin = LogisticRegression().fit(X[tr][:, :2], y[tr])
        oof_lin[te] = lin.predict_proba(X[te][:, :2])[:, 1]
        bar.update(k + 1)
    bar.close()

    results = {}
    for name, s in (("linear blend (v2.0)", oof_lin), ("level-2 stacker", oof_stack)):
        pairs = val_all.merge(d[KEY].assign(new=s), on=KEY, how="left")
        pairs["score"] = pairs["new"].fillna(pairs["score"])
        thr = tune_threshold(codes_all, pairs["score"].to_numpy(), pairs["label"].to_numpy(), truth["n_true"].to_numpy(), val_mask).threshold
        results[name] = _macro(pairs, thr, val_ent)
        results[name]["threshold"] = thr
    base_thr = json.loads((config.MODELS_DIR / val_run / "metrics.json").read_text())["threshold"]
    results["GBDT only"] = _macro(val_all, base_thr, val_ent)
    print(pd.DataFrame(results)[["GBDT only", "linear blend (v2.0)", "level-2 stacker"]].to_string(float_format=lambda x: f"{x:.4f}"))
    (CE_DIR / f"stack2_cv{tag}.json").write_text(json.dumps({k: float(v) for k, v in results["level-2 stacker"].items()}, indent=2))


def apply(val_run: str, test_run: str, out_dir: str, tag: str = "") -> None:
    from export import export_submission
    from io_utils import read_cleaned_table
    from predict import summarize

    _stage(f"fitting the stacker on all dev_val pairs (CE scores{tag or ' round 1'})")
    d = _val_frame(val_run, tag)
    booster = xgb.train(PARAMS, xgb.DMatrix(_design(d), label=d["label"].to_numpy()), ROUNDS)
    val_all = pd.read_parquet(config.MODELS_DIR / val_run / "val_predictions.parquet")
    truth = entity_truth(load_ground_truth())
    val_mask = split_entities(truth["country"].to_numpy()) == ROLE_VAL
    # threshold from the CV protocol would need the OOF run; in-sample stacker scores on
    # dev_val would overstate confidence, so reuse CV: fit on one fold, tune on the other
    fold = (pd.util.hash_pandas_object(d["source1_entity_id"], index=False).to_numpy() % 2).astype(int)
    oof = np.zeros(len(d))
    for k in (0, 1):
        b = xgb.train(PARAMS, xgb.DMatrix(_design(d[fold != k]), label=d["label"].to_numpy()[fold != k]), ROUNDS)
        oof[fold == k] = b.predict(xgb.DMatrix(_design(d[fold == k])))
    pairs = val_all.merge(d[KEY].assign(new=oof), on=KEY, how="left")
    pairs["score"] = pairs["new"].fillna(pairs["score"])
    codes = pd.Index(truth["source1_entity_id"]).get_indexer(pairs["source1_entity_id"])
    thr = tune_threshold(codes, pairs["score"].to_numpy(), pairs["label"].to_numpy(), truth["n_true"].to_numpy(), val_mask).threshold
    print(f"  stacker threshold (from CV): {thr:.3f}", flush=True)

    _stage("scoring test pairs")
    tce = _attach_features(pd.read_parquet(CE_DIR / f"test_scores{tag}.parquet"), config.FEATURES_DIR / f"test_{UNION}")
    tce["new"] = booster.predict(xgb.DMatrix(_design(tce)))
    test_all = pd.read_parquet(config.PREDICTIONS_DIR / test_run / f"test_{UNION}.parquet")
    test_all = test_all.merge(tce[KEY + ["new"]], on=KEY, how="left")
    test_all["score"] = test_all["new"].fillna(test_all["score"]).astype(np.float32)
    test_all = test_all.drop(columns="new")

    _stage(f"writing submission to {out_dir}")
    matches = select_matches(test_all, thr)
    source1 = read_cleaned_table("test", "source1", columns=["entity_id", "country"]).to_pandas()
    export_submission(Path(out_dir), source1["entity_id"].to_numpy(), test_all, matches)
    print(summarize(source1, test_all, matches).to_string(float_format=lambda x: f"{x:.4f}"))
    booster.save_model(str(CE_DIR / f"stack2_model{tag}.json"))
    (CE_DIR / f"stack2_config{tag}.json").write_text(json.dumps({"threshold": thr, "ce_tag": tag, "rounds": ROUNDS, "params": PARAMS,
                                                          "features": PAIR_FEATURES}, indent=2))
    _stage("done -- validate, then upload")


def main() -> None:
    parser = argparse.ArgumentParser(description="Level-2 stacker over GBDT + cross-encoder scores.")
    parser.add_argument("cmd", choices=["cv", "apply"])
    parser.add_argument("--val-run", default="dev_v11")
    parser.add_argument("--test-run", default="final_v11")
    parser.add_argument("--out-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--tag", default="", help="Which CE scores: '' = round 1, '_r2' = round 2.")
    args = parser.parse_args()
    if args.cmd == "cv":
        cv(args.val_run, args.tag)
    else:
        apply(args.val_run, args.test_run, args.out_dir, args.tag)


if __name__ == "__main__":
    main()
