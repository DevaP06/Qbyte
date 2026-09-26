"""Every wrong decision a dev run makes on its dev_val entities, as one file.

    python error_analysis.py --run dev_union
    -> data/processed/errors/<run>_errors.parquet  (+ .tsv for spreadsheets)

error_type:
  FP    predicted match, not a true match (after threshold + conflict resolution)
  FN    true match that WAS a candidate but ended below the threshold / lost a conflict
  MISS  true match that never became a candidate (blocking/retrieval miss)

Each row carries both records' texts, the score, and the entity's F0.5 and
loss (1 - F0.5, what the entity costs the macro average), so the file sorts
straight into "biggest problems first". dev_val entities were never trained
on, so these are honest errors.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from evaluate import per_entity_f05
from finetune_data import record_texts
from labels import entity_truth, ground_truth_pairs, load_ground_truth
from postprocess import select_matches
from train_classifier import ROLE_VAL, split_entities

ERRORS_DIR = config.DATA_PROCESSED_DIR / "errors"


def build_errors(run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = config.MODELS_DIR / run
    threshold = float(pd.read_json(run_dir / "metrics.json", typ="series")["threshold"])
    vp = pd.read_parquet(run_dir / "val_predictions.parquet")

    gt = load_ground_truth()
    truth = entity_truth(gt)
    val = truth[split_entities(truth["country"].to_numpy()) == ROLE_VAL].set_index("source1_entity_id")
    pairs = ground_truth_pairs(gt)
    pairs = pairs[pairs["source1_entity_id"].isin(val.index)]

    key = lambda a, b: a.astype(str) + "|" + b.astype(str)
    matched = select_matches(vp, threshold)
    vp["pred"] = key(vp["source1_entity_id"], vp["candidate_entity_id"]).isin(
        set(key(matched["source1_entity_id"], matched["candidate_entity_id"]))
    )
    fp = vp[vp["pred"] & (vp["label"] == 0)].assign(error_type="FP")
    fn = vp[~vp["pred"] & (vp["label"] == 1)].assign(error_type="FN")
    in_cands = key(pairs["source1_entity_id"], pairs["matched_id"]).isin(
        set(key(vp["source1_entity_id"], vp["candidate_entity_id"]))
    )
    miss = pairs[~in_cands.to_numpy()].rename(columns={"matched_id": "candidate_entity_id"}).assign(
        error_type="MISS", score=np.nan, label=1
    )
    errors = pd.concat([fp, fn, miss], ignore_index=True)[
        ["error_type", "source1_entity_id", "candidate_entity_id", "score", "label"]
    ]

    # Entity-level F0.5 exactly as scored (all val entities, candidate-less ones included).
    tp = vp[vp["pred"] & (vp["label"] == 1)].groupby("source1_entity_id").size().reindex(val.index).fillna(0)
    n_pred = vp[vp["pred"]].groupby("source1_entity_id").size().reindex(val.index).fillna(0)
    val["f05"] = per_entity_f05(tp.to_numpy(float), n_pred.to_numpy(float), val["n_true"].to_numpy(float))
    val["n_pred"] = n_pred.astype(int)

    texts = record_texts("train")
    errors["country"] = val["country"].reindex(errors["source1_entity_id"]).to_numpy()
    errors["n_true"] = val["n_true"].reindex(errors["source1_entity_id"]).to_numpy()
    errors["entity_f05"] = val["f05"].reindex(errors["source1_entity_id"]).to_numpy()
    errors["entity_loss"] = 1 - errors["entity_f05"]
    errors["s1_text"] = texts.reindex(errors["source1_entity_id"]).to_numpy()
    errors["candidate_text"] = texts.reindex(errors["candidate_entity_id"]).to_numpy()
    errors = errors.sort_values(["entity_loss", "source1_entity_id", "error_type"], ascending=[False, True, True])
    return errors.reset_index(drop=True), val


def summary(errors: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Macro-F0.5 points lost, split by the entity's error kinds."""
    kinds = errors.groupby("source1_entity_id")["error_type"].agg(lambda s: "+".join(sorted(set(s))))
    v = val.assign(kinds=kinds.reindex(val.index).fillna("none"), loss=1 - val["f05"])
    v.loc[(v["n_true"] == 0) & (v["kinds"] == "FP"), "kinds"] = "FP (true singleton)"
    out = v.groupby(["country", "kinds"]).agg(entities=("loss", "size"), loss_points=("loss", "sum"))
    out["loss_points"] = out["loss_points"] / len(v)
    return out[out.index.get_level_values("kinds") != "none"].sort_values("loss_points", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write every wrong dev_val decision of a dev run to one file.")
    parser.add_argument("--run", default="dev_union")
    args = parser.parse_args()

    errors, val = build_errors(args.run)
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    base = ERRORS_DIR / f"{args.run}_errors"
    errors.to_parquet(base.with_suffix(".parquet"), index=False)
    errors.to_csv(base.with_suffix(".tsv"), sep="\t", index=False, encoding="utf-8", lineterminator="\n")

    pd.set_option("display.width", 200)
    print(f"dev_val macro F0.5 {val['f05'].mean():.4f} over {len(val):,} entities; "
          f"{len(errors):,} wrong pairs -> {base}.parquet/.tsv")
    print(errors["error_type"].value_counts().to_string())
    print("\nwhere the macro F0.5 is lost:")
    print(summary(errors, val).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
