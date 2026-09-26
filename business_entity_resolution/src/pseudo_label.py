"""Unseen-country adaptation by pseudo-labeling -- simulated on train before it is
applied to France.

France is in test only; on the portal it scores ~0.927 against ~0.986 for
India/US, with twice their share of uncertain candidates. Pseudo-labeling
retrains on the model's own *confident* decisions for the unseen country
(score >= --pos as match, <= --neg as non-match, the uncertain middle left
out), so the classifier adapts to that country's feature distribution. It can
also entrench mistakes, so this script measures it first, pretending a
labeled country is unseen:

    python pseudo_label.py --simulate US --fit-entity-frac 0.3

  A  train on every country except the target (labels) -> score target dev_val
  B  score the target's fit entities with A, keep confident pseudo-labels,
     retrain on source labels + target pseudo-labels -> score target dev_val
Thresholds come from the source countries' dev_val only, exactly as France's
must. Reported: A vs B on the target, plus how accurate the pseudo-labels
were (known here, unknowable for France). B > A is the go/no-go for France.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import config
from labels import PairLabeler, entity_truth, ground_truth_pairs, load_ground_truth
from train_classifier import (
    ROLE_EARLY_STOP,
    ROLE_FIT,
    ROLE_VAL,
    XGBoostTrainer,
    _fmt_duration,
    _rows_for,
    _stage,
    _subsample_entities,
    evaluate_on,
    load_training_data,
    split_entities,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate pseudo-label adaptation to an unseen country.")
    parser.add_argument("--simulate", default="US", help="Train country treated as unseen (France stand-in).")
    parser.add_argument("--features", default="train_union_retriever_k20")
    parser.add_argument("--fit-entity-frac", type=float, default=0.3, help="Fit entities used, for speed.")
    parser.add_argument("--pos", type=float, default=0.98, help="Score >= this -> pseudo match.")
    parser.add_argument("--neg", type=float, default=0.02, help="Score <= this -> pseudo non-match.")
    args = parser.parse_args()
    t_start = time.time()

    _stage(f"loading features/{args.features}")
    gt = load_ground_truth()
    truth = entity_truth(gt)
    data = load_training_data(config.FEATURES_DIR / args.features, truth, PairLabeler(ground_truth_pairs(gt)))
    countries = truth["country"].to_numpy()
    roles = split_entities(countries)
    target = countries == args.simulate
    fit = _subsample_entities(roles == ROLE_FIT, args.fit_entity_frac, config.SPLIT_SEED)
    src_fit, src_es, src_val = _rows_for(data, fit & ~target), _rows_for(data, (roles == ROLE_EARLY_STOP) & ~target), (roles == ROLE_VAL) & ~target
    tgt_fit_rows, tgt_val = _rows_for(data, fit & target), (roles == ROLE_VAL) & target
    tgt_val_rows, src_val_rows = _rows_for(data, tgt_val), _rows_for(data, src_val)
    trainer = XGBoostTrainer(data, device="cuda")

    def rate_matched_threshold(src_scores, thr, tgt_scores) -> float:
        """Label-free: the target threshold at which the target accepts as many
        pairs per entity as the source does at its tuned threshold."""
        rate = (src_scores >= thr).sum() / src_val.sum()
        n_accept = int(round(rate * tgt_val.sum()))
        return float(np.sort(tgt_scores)[::-1][max(n_accept - 1, 0)])

    def score_target(model, label):
        src_scores, tgt_scores = model.predict(data.X[src_val_rows]), model.predict(data.X[tgt_val_rows])
        thr = evaluate_on(data, src_scores, src_val_rows, src_val, label="threshold (source)").threshold
        rep = evaluate_on(data, tgt_scores, tgt_val_rows, tgt_val, threshold=thr)
        thr_rate = rate_matched_threshold(src_scores, thr, tgt_scores)
        rep_rate = evaluate_on(data, tgt_scores, tgt_val_rows, tgt_val, threshold=thr_rate)
        print(f"  {label}: {args.simulate} dev_val macro F0.5 = {rep.f05_by_country[args.simulate]:.4f} (source threshold {thr:.2f}) | "
              f"rate-matched threshold {thr_rate:.2f}: {rep_rate.f05_by_country[args.simulate]:.4f}", flush=True)
        return rep.f05_by_country[args.simulate]

    _stage(f"A: train without {args.simulate} ({len(src_fit):,} source pairs)")
    model_a = trainer.fit(src_fit, src_es, label="A: source only")
    f_a = score_target(model_a, "A (unseen, no adaptation)")

    _stage(f"B: pseudo-label {len(tgt_fit_rows):,} {args.simulate} fit pairs with A")
    s = model_a.predict(data.X[tgt_fit_rows])
    confident = (s >= args.pos) | (s <= args.neg)
    pseudo = (s >= args.pos).astype(np.int8)
    true = data.y[tgt_fit_rows]
    kept = tgt_fit_rows[confident]
    print(f"  kept {confident.mean() * 100:.1f}% as pseudo-labels; accuracy vs truth {np.mean(pseudo[confident] == true[confident]) * 100:.2f}% "
          f"(pseudo-positives correct: {true[confident & (pseudo == 1)].mean() * 100:.2f}%, "
          f"true matches recovered as positives: {pseudo[true == 1].mean() * 100:.1f}%)", flush=True)

    trainer.y = data.y.copy()
    trainer.y[kept] = pseudo[confident]
    model_b = trainer.fit(np.sort(np.concatenate([src_fit, kept])), src_es, label="B: + pseudo")
    f_b = score_target(model_b, "B (unseen + pseudo-labels)")

    _stage(f"result for unseen {args.simulate}: A {f_a:.4f} -> B {f_b:.4f} ({f_b - f_a:+.4f})  "
           f"[{'GO: use for France' if f_b > f_a else 'NO-GO'}]  total {_fmt_duration(time.time() - t_start)}")


if __name__ == "__main__":
    main()
