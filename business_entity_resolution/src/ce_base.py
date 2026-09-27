"""Larger cross-encoder: multilingual-e5-base (MIT, 278M) for v2.5.

    python ce_base.py benchmark --budget-min 150     # GPU: time 200 steps -> how many pairs fit the budget
    python ce_base.py export --pairs N               # CPU: training pairs from CE rounds 1-3
    python ce_base.py score                          # GPU: score only the uncertain pairs

Training itself reuses cross_encoder.py train with --base-model
intfloat/multilingual-e5-base (a new classification head on the base model).

Why: the small CE's errors are text judgments (substituted business word,
another street at the same number, degraded variants). A 2.4x larger model
is the one lever that attacks model strength in every country. To fit the
time left it trains on a budget-sized sample, and scores only pairs that the
GBDT and the round-3 CE do not both call confident matches (>= 0.98); the
stacker sees NaN elsewhere.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import config
from cross_encoder import CE_DIR, _assert_no_dev_val
from finetune_data import record_texts
from labels import entity_truth, load_ground_truth
from train_classifier import Progress, _stage, split_entities

BASE_MODEL = "intfloat/multilingual-e5-base"
BASE_DIR = config.MODELS_DIR / "cross_encoder_base"
KEY = ["source1_entity_id", "candidate_entity_id"]


def benchmark(budget_min: float, batch_size: int) -> None:
    """Time 200 training steps of e5-base; write the number of training pairs
    that fits `budget_min` minutes to ce_base_budget.json."""
    import torch
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    src = CE_DIR / ("train_base.parquet" if (CE_DIR / "train_base.parquet").exists() else "train_r3.parquet")
    tr = pd.read_parquet(src).sample(200 * batch_size, random_state=0)
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=128)
    loader = DataLoader([InputExample(texts=[a, b], label=float(y)) for a, b, y in zip(tr["text_a"], tr["text_b"], tr["label"])],
                        shuffle=True, batch_size=batch_size)
    t0 = time.time()
    model.fit(train_dataloader=loader, epochs=1, warmup_steps=20, use_amp=torch.cuda.is_available(), show_progress_bar=True)
    sec_per_step = (time.time() - t0) / 200
    pairs = int(budget_min * 60 / sec_per_step * batch_size)
    (CE_DIR / "ce_base_budget.json").write_text(json.dumps({"sec_per_step": sec_per_step, "budget_min": budget_min, "pairs": pairs}, indent=2))
    print(f"  e5-base: {sec_per_step:.3f} s/step at batch {batch_size} -> {pairs:,} training pairs fit {budget_min:.0f} min", flush=True)


def export(n_pairs: int, france_share: float, seed: int) -> None:
    """Budget-sized sample of CE rounds 1-3. France rows (unlabeled-test
    agreement / synthetic / S1-S1 pairs) get `france_share` of the sample; the
    rest is India/US labelled pairs. French agreement pairs are refreshed from
    the round-3 CE when its test scores exist."""
    rng_seed = seed + 21
    parts = [pd.read_parquet(CE_DIR / f) for f in ("train.parquet", "train_r2.parquet", "train_r3.parquet") if (CE_DIR / f).exists()]
    allp = pd.concat(parts, ignore_index=True).drop_duplicates(["source1_entity_id", "candidate_entity_id", "text_b"])
    truth = entity_truth(load_ground_truth())
    roles = split_entities(truth["country"].to_numpy())
    train_s1 = set(truth["source1_entity_id"])
    labelled = allp[allp["source1_entity_id"].isin(train_s1)]
    france = allp[~allp["source1_entity_id"].isin(train_s1)]
    _assert_no_dev_val(labelled, truth, roles)

    r3_scores = CE_DIR / "test_scores_r3.parquet"
    if r3_scores.exists():
        _stage("refreshing French agreement pairs from GBDT + round-3 CE")
        ts = pd.read_parquet(r3_scores)
        fr = ts[ts["country"] == "France"]
        pos = fr[(fr["score"] >= 0.98) & (fr["ce"] >= 0.98)].sort_values("ce", ascending=False).drop_duplicates("candidate_entity_id")
        neg = fr[(fr["ce"] <= 0.02) & (fr["score"] < 0.5)]
        k = 200_000
        agree = pd.concat([pos.sample(min(k, len(pos)), random_state=seed).assign(label=1),
                           neg.sample(min(k, len(neg)), random_state=seed).assign(label=0)])
        texts = record_texts("test")
        agree = agree.assign(text_a=texts.reindex(agree["source1_entity_id"]).to_numpy(),
                             text_b=texts.reindex(agree["candidate_entity_id"]).to_numpy())[france.columns]
        synthetic = france[france["candidate_entity_id"].str.startswith("synthetic") | france["candidate_entity_id"].str.startswith("S1-")]
        france = pd.concat([agree, synthetic], ignore_index=True)

    n_fr = min(len(france), int(n_pairs * france_share))
    n_lab = min(len(labelled), n_pairs - n_fr)
    out = pd.concat([france.sample(n_fr, random_state=rng_seed), labelled.sample(n_lab, random_state=rng_seed)]).sample(frac=1.0, random_state=rng_seed)
    out.to_parquet(CE_DIR / "train_base.parquet", index=False)
    print(f"  e5-base train {len(out):,} pairs: {n_lab:,} India/US labelled + {n_fr:,} France "
          f"(agreement/synthetic/S1-S1); positive share {out['label'].mean() * 100:.1f}%", flush=True)


def score(hi: float, batch_size: int, limit: int) -> None:
    """e5-base probability for pairs NOT confidently matched by both the GBDT
    and the round-3 CE; written as val_scores_base / test_scores_base (the
    stacker treats unscored pairs as missing)."""
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder(str(BASE_DIR), max_length=128)
    if torch.cuda.is_available():
        model.model.half()
    for kind, split in (("val", "train"), ("test", "test")):
        src = pd.read_parquet(CE_DIR / f"{kind}_scores_r3.parquet")
        sel = src[~((src["score"] >= hi) & (src["ce"] >= hi))].drop(columns=["ce"])
        if limit:
            sel = sel.head(limit)
        texts = record_texts(split)
        a = texts.reindex(sel["source1_entity_id"]).fillna("").to_numpy()
        b = texts.reindex(sel["candidate_entity_id"]).fillna("").to_numpy()
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])  # similar lengths per batch -> less padding
        _stage(f"e5-base scoring {len(sel):,} uncertain {kind} pairs (of {len(src):,})")
        pred = model.predict(list(zip(a[order], b[order])), batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
        ce = np.empty(len(sel), dtype=np.float32)
        ce[order] = pred.float().cpu().numpy()
        sel.assign(ce=ce).to_parquet(CE_DIR / f"{kind}_scores_base.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="multilingual-e5-base cross-encoder for v2.5.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--budget-min", type=float, default=150)
    b.add_argument("--batch-size", type=int, default=128)
    e = sub.add_parser("export")
    e.add_argument("--pairs", type=int, default=0, help="0 = read from ce_base_budget.json")
    e.add_argument("--france-share", type=float, default=0.35)
    e.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    s = sub.add_parser("score")
    s.add_argument("--hi", type=float, default=0.98)
    s.add_argument("--batch-size", type=int, default=512)
    s.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.cmd == "benchmark":
        benchmark(args.budget_min, args.batch_size)
    elif args.cmd == "export":
        n = args.pairs or json.loads((CE_DIR / "ce_base_budget.json").read_text())["pairs"]
        export(n, args.france_share, args.seed)
    else:
        score(args.hi, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
