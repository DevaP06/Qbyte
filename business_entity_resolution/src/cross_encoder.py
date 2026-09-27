"""Cross-encoder re-scoring of borderline pairs (cycle 2).

Why: the remaining errors -- above all France's -- are text-level: the same
address with one business word substituted ("lutins institut sas" vs "lutins
comite sas"), a different street at the same house number, acronyms. Pairwise
GBDT features barely register a one-word substitution; a cross-encoder reads
both records together. It starts from our fine-tuned retriever (multilingual
e5-small, MIT), so it already knows these records and carries over to French.

    python cross_encoder.py export     # CPU: training pairs from fit entities (dev_val never used)
    python cross_encoder.py train      # GPU: fine-tune (progress bar), ~30-60 min
    python cross_encoder.py score      # GPU: score borderline dev_val + test pairs
    python cross_encoder.py blend      # CPU: tune GBDT/CE blend on dev_val, write the blended test scores

Only pairs the GBDT is unsure about (score in [--lo, --hi]) are re-scored;
confident decisions stay as they are.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from finetune_data import record_texts
from labels import PairLabeler, entity_truth, ground_truth_pairs, load_ground_truth
from train_classifier import ROLE_EARLY_STOP, ROLE_FIT, Progress, _stage, split_entities

CE_DIR = config.DATA_PROCESSED_DIR / "cross_encoder"
CE_MODEL_DIR = config.MODELS_DIR / "cross_encoder"
UNION = "union_retriever_k20"


def _pairs_for(features_dir: Path, s1_ids: set, columns: list) -> pd.DataFrame:
    """Rows of a features dataset whose S1 is in `s1_ids` (streamed part by part)."""
    want = pa.array(list(s1_ids))
    parts = sorted(features_dir.glob("part-*.parquet"))
    bar = Progress(len(parts), "reading features", "parts")
    out = []
    for i, p in enumerate(parts, start=1):
        t = pq.read_table(p, columns=columns)
        out.append(t.filter(pc.is_in(t.column("source1_entity_id"), value_set=want)).to_pandas())
        bar.update(i)
    bar.close()
    return pd.concat(out, ignore_index=True)


def _labelled_pairs(s1_ids: set, gt: pd.DataFrame, negatives_per_entity: int, seed: int) -> pd.DataFrame:
    """All true pairs + the hardest negatives (highest emb_cos + name similarity)
    of the given train S1 entities, with both records' texts."""
    _stage(f"selecting pairs for {len(s1_ids):,} train entities")
    f = _pairs_for(config.FEATURES_DIR / f"train_{UNION}", s1_ids,
                   ["source1_entity_id", "candidate_entity_id", "emb_cos", "name_token_set_ratio"])
    f["label"] = PairLabeler(ground_truth_pairs(gt)).label(f["source1_entity_id"].to_numpy(), f["candidate_entity_id"].to_numpy())
    f["hardness"] = f["emb_cos"].fillna(0) + f["name_token_set_ratio"].fillna(0)
    neg = f[f["label"] == 0].sort_values("hardness", ascending=False).groupby("source1_entity_id").head(negatives_per_entity)
    pairs = pd.concat([f[f["label"] == 1], neg])
    _stage("attaching record texts")
    texts = record_texts("train")
    return pairs.assign(
        text_a=texts.reindex(pairs["source1_entity_id"]).to_numpy(),
        text_b=texts.reindex(pairs["candidate_entity_id"]).to_numpy(),
    )[["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "label"]].sample(frac=1.0, random_state=seed)


def export_round2(n_entities: int, france_per_class: int, negatives_per_entity: int, seed: int) -> None:
    """Round-2 training data: labelled pairs of fit entities round 1 never saw,
    plus France agreement pairs -- test pairs where the GBDT and the round-1 CE
    agree strongly (both >= 0.98: match; CE <= 0.02 with GBDT < 0.5: non-match).
    Those are the most reliable labels available for France (two different
    models, text vs engineered features, agreeing) and teach the CE French
    street/name patterns. dev_val entities are never included."""
    rng = np.random.default_rng(seed + 1)
    gt = load_ground_truth()
    truth = entity_truth(gt)
    roles = split_entities(truth["country"].to_numpy())
    used = set(pd.read_parquet(CE_DIR / "train.parquet", columns=["source1_entity_id"])["source1_entity_id"])
    fresh = np.array([i for i in truth["source1_entity_id"].to_numpy()[roles == ROLE_FIT] if i not in used])
    pairs = _labelled_pairs(set(rng.choice(fresh, size=min(n_entities, len(fresh)), replace=False)), gt, negatives_per_entity, seed)

    _stage("France agreement pairs from test")
    ts = pd.read_parquet(CE_DIR / "test_scores.parquet")
    fr = ts[ts["country"] == "France"]
    pos = fr[(fr["score"] >= 0.98) & (fr["ce"] >= 0.98)]
    neg = fr[(fr["ce"] <= 0.02) & (fr["score"] < 0.5)]
    pos = pos.sample(min(france_per_class, len(pos)), random_state=seed).assign(label=1)
    neg = neg.sample(min(france_per_class, len(neg)), random_state=seed).assign(label=0)
    texts = record_texts("test")
    france = pd.concat([pos, neg])
    france = france.assign(
        text_a=texts.reindex(france["source1_entity_id"]).to_numpy(),
        text_b=texts.reindex(france["candidate_entity_id"]).to_numpy(),
    )[["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "label"]]
    out = pd.concat([pairs, france]).sample(frac=1.0, random_state=seed)
    out.to_parquet(CE_DIR / "train_r2.parquet", index=False)
    print(f"  round-2 train: {len(pairs):,} labelled pairs from {min(n_entities, len(fresh)):,} new fit entities "
          f"+ {len(france):,} France agreement pairs ({len(pos):,} match / {len(neg):,} non-match) -> train_r2.parquet", flush=True)


def export(n_entities: int, negatives_per_entity: int, seed: int) -> None:
    """Positives + the most confusable negatives (highest emb_cos / name
    similarity) of sampled fit entities; eval pairs from early-stop entities."""
    rng = np.random.default_rng(seed)
    gt = load_ground_truth()
    truth = entity_truth(gt)
    roles = split_entities(truth["country"].to_numpy())
    ids = truth["source1_entity_id"].to_numpy()
    fit_ids = set(rng.choice(ids[roles == ROLE_FIT], size=n_entities, replace=False))
    es_ids = set(rng.choice(ids[roles == ROLE_EARLY_STOP], size=min(20_000, (roles == ROLE_EARLY_STOP).sum()), replace=False))

    pairs = _labelled_pairs(fit_ids | es_ids, gt, negatives_per_entity, seed)
    CE_DIR.mkdir(parents=True, exist_ok=True)
    is_eval = pairs["source1_entity_id"].isin(es_ids)
    pairs[~is_eval].to_parquet(CE_DIR / "train.parquet", index=False)
    pairs[is_eval].to_parquet(CE_DIR / "eval.parquet", index=False)
    print(f"  train {int((~is_eval).sum()):,} pairs ({pairs.loc[~is_eval, 'label'].mean() * 100:.1f}% positive), "
          f"eval {int(is_eval.sum()):,} -> {CE_DIR}", flush=True)


def train(base_model: str, epochs: float, batch_size: int, lr: float, max_length: int, max_rows: int,
          train_file: str = "train.parquet", out_dir: Path = CE_MODEL_DIR) -> None:
    import torch
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    print(f"device: {'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    tr = pd.read_parquet(CE_DIR / train_file)
    ev = pd.read_parquet(CE_DIR / "eval.parquet")
    ev = ev.sample(min(len(ev), 30_000), random_state=0)  # each periodic eval re-scores this set
    if max_rows:
        tr = tr.head(max_rows)
    print(f"[1/3] {len(tr):,} training pairs, {len(ev):,} eval pairs; base model {base_model}", flush=True)
    model = CrossEncoder(base_model, num_labels=1, max_length=max_length)
    examples = [InputExample(texts=[a, b], label=float(y)) for a, b, y in zip(tr["text_a"], tr["text_b"], tr["label"])]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    evaluator = CEBinaryClassificationEvaluator(list(zip(ev["text_a"], ev["text_b"])), ev["label"].tolist(), name="eval")
    steps = int(len(loader) * epochs)
    print(f"[2/3] training {steps:,} steps (progress bar below; eval every {max(steps // 5, 1):,} steps)", flush=True)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=loader, evaluator=evaluator, epochs=max(1, round(epochs)), warmup_steps=int(0.1 * steps),
        optimizer_params={"lr": lr}, evaluation_steps=max(steps // 5, 1), use_amp=torch.cuda.is_available(),
        show_progress_bar=True,
    )
    print("[3/3] final evaluation ...", flush=True)
    score = evaluator(model)
    model.save(str(out_dir))
    (out_dir / "ce_config.json").write_text(json.dumps(
        {"base_model": base_model, "train_file": train_file, "train_rows": len(tr), "epochs": epochs, "batch_size": batch_size,
         "lr": lr, "max_length": max_length, "eval_average_precision": score, "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"eval average precision {score:.4f}; saved to {out_dir} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _score_file(model, split: str, src: Path, dest: Path, lo: float, batch_size: int, limit: int) -> None:
    df = pd.read_parquet(src)
    sel = df[df["score"] >= lo]
    if limit:
        sel = sel.head(limit)
    texts = record_texts(split)
    pairs = list(zip(texts.reindex(sel["source1_entity_id"]).to_numpy(), texts.reindex(sel["candidate_entity_id"]).to_numpy()))
    print(f"  cross-encoding {len(pairs):,} {split} pairs (GBDT score >= {lo}) -> {dest.name}", flush=True)
    # convert_to_tensor stacks once; convert_to_numpy converts millions of per-row tensors one by one (very slow)
    ce = model.predict(pairs, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    sel = sel.assign(ce=ce.float().cpu().numpy())
    dest.parent.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(dest, index=False)


def score(val_run: str, test_run: str, lo: float, batch_size: int, limit: int,
          model_dir: Path = CE_MODEL_DIR, tag: str = "") -> None:
    """CE probability for every pair the GBDT gives >= lo: dev_val pairs (to
    tune the blend) and test pairs (to apply it). Confident accepts are
    included on purpose -- France's worst errors score 0.98+. Writes
    val_scores{tag}.parquet / test_scores{tag}.parquet."""
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder(str(model_dir), max_length=128)
    if torch.cuda.is_available():
        model.model.half()
    _stage(f"scoring dev_val pairs with {model_dir.name}")
    _score_file(model, "train", config.MODELS_DIR / val_run / "val_predictions.parquet", CE_DIR / f"val_scores{tag}.parquet", lo, batch_size, limit)
    _stage(f"scoring test pairs with {model_dir.name}")
    _score_file(model, "test", config.PREDICTIONS_DIR / test_run / f"test_{UNION}.parquet", CE_DIR / f"test_scores{tag}.parquet", lo, batch_size, limit)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(val_run: str, test_run: str, out_dir: str) -> None:
    """Fit logit(final) = a*logit(gbdt) + b*logit(ce) + c on dev_val pairs the
    CE scored, tune the macro-F0.5 threshold on dev_val, report the gain over
    GBDT alone, then apply the same blend + threshold to test and write the
    submission files."""
    from sklearn.linear_model import LogisticRegression

    from evaluate import per_entity_f05
    from export import export_submission
    from io_utils import read_cleaned_table
    from postprocess import select_matches
    from predict import summarize
    from threshold_tuning import tune_threshold
    from train_classifier import ROLE_VAL

    val_all = pd.read_parquet(config.MODELS_DIR / val_run / "val_predictions.parquet")
    vce = pd.read_parquet(CE_DIR / "val_scores.parquet")
    lr = LogisticRegression(C=1.0)
    lr.fit(np.column_stack([_logit(vce["score"].to_numpy()), _logit(vce["ce"].to_numpy())]), vce["label"].to_numpy())

    def apply(all_pairs: pd.DataFrame, ce_pairs: pd.DataFrame) -> pd.DataFrame:
        blended = lr.predict_proba(np.column_stack([_logit(ce_pairs["score"].to_numpy()), _logit(ce_pairs["ce"].to_numpy())]))[:, 1]
        key = ["source1_entity_id", "candidate_entity_id"]
        out = all_pairs.merge(ce_pairs[key].assign(blended=blended), on=key, how="left")
        out["score"] = out["blended"].fillna(out["score"]).astype(np.float32)  # unscored pairs (< lo) keep the GBDT score
        return out.drop(columns="blended")

    truth = entity_truth(load_ground_truth())
    val_mask = split_entities(truth["country"].to_numpy()) == ROLE_VAL
    val_ent = truth[val_mask].set_index("source1_entity_id")
    codes = pd.Index(truth["source1_entity_id"]).get_indexer(val_all["source1_entity_id"])
    n_true = truth["n_true"].to_numpy()

    def macro(pairs: pd.DataFrame, thr: float) -> pd.Series:
        m = select_matches(pairs, thr)
        tp = m[m["label"] == 1].groupby("source1_entity_id").size().reindex(val_ent.index).fillna(0)
        n = m.groupby("source1_entity_id").size().reindex(val_ent.index).fillna(0)
        f = pd.Series(per_entity_f05(tp.to_numpy(float), n.to_numpy(float), val_ent["n_true"].to_numpy(float)), index=val_ent.index)
        s = f.groupby(val_ent["country"]).mean()
        s["ALL"] = f.mean()
        return s

    base_thr = json.loads((config.MODELS_DIR / val_run / "metrics.json").read_text())["threshold"]
    val_blend = apply(val_all, vce)
    thr = tune_threshold(codes, val_blend["score"].to_numpy(), val_blend["label"].to_numpy(), n_true, val_mask).threshold
    before, after = macro(val_all, base_thr), macro(val_blend, thr)
    print(pd.DataFrame({"GBDT only": before, "GBDT + cross-encoder": after, "gain": after - before}).to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 0.5 else f"{x:.4f}"))
    print(f"  blend: {lr.coef_[0][0]:.3f}*logit(gbdt) + {lr.coef_[0][1]:.3f}*logit(ce) + {lr.intercept_[0]:.3f}; threshold {thr:.3f}", flush=True)

    _stage("applying the blend to test and writing the submission")
    test_all = pd.read_parquet(config.PREDICTIONS_DIR / test_run / f"test_{UNION}.parquet")
    test_blend = apply(test_all, pd.read_parquet(CE_DIR / "test_scores.parquet"))
    matches = select_matches(test_blend, thr)
    source1 = read_cleaned_table("test", "source1", columns=["entity_id", "country"]).to_pandas()
    export_submission(Path(out_dir), source1["entity_id"].to_numpy(), test_blend, matches)
    print(summarize(source1, test_blend, matches).to_string(float_format=lambda x: f"{x:.4f}"))
    (CE_DIR / "blend_config.json").write_text(json.dumps(
        {"coef_gbdt": lr.coef_[0][0], "coef_ce": lr.coef_[0][1], "intercept": lr.intercept_[0], "threshold": thr,
         "val_run": val_run, "test_run": test_run, "dev_val_before": before.to_dict(), "dev_val_after": after.to_dict()}, indent=2))
    _stage(f"done -- submission files in {out_dir}; validate, then upload")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-encoder re-scoring of borderline pairs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--entities", type=int, default=400_000, help="Fit entities sampled for training pairs.")
    e.add_argument("--negatives", type=int, default=4, help="Hardest negatives kept per entity.")
    e.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    e.add_argument("--round2", action="store_true", help="New fit entities + France agreement pairs -> train_r2.parquet.")
    e.add_argument("--france-per-class", type=int, default=200_000)
    t = sub.add_parser("train")
    t.add_argument("--base-model", default=str(config.MODELS_DIR / "retriever"))
    t.add_argument("--train-file", default="train.parquet")
    t.add_argument("--out-model", default=str(CE_MODEL_DIR))
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--max-length", type=int, default=128)
    t.add_argument("--max-rows", type=int, default=0, help="0 = all (use a small number for a smoke test).")
    s = sub.add_parser("score")
    s.add_argument("--val-run", default="dev_v11")
    s.add_argument("--test-run", default="final_v11")
    s.add_argument("--lo", type=float, default=0.02, help="Re-score pairs with GBDT score >= this.")
    s.add_argument("--batch-size", type=int, default=512)
    s.add_argument("--limit", type=int, default=0, help="Score only the first N pairs per split (smoke test).")
    s.add_argument("--model", default=str(CE_MODEL_DIR))
    s.add_argument("--tag", default="", help="Output suffix: val_scores{tag}.parquet / test_scores{tag}.parquet.")
    b = sub.add_parser("blend")
    b.add_argument("--val-run", default="dev_v11")
    b.add_argument("--test-run", default="final_v11")
    b.add_argument("--out-dir", default=str(config.OUTPUT_DIR))
    args = parser.parse_args()
    if args.cmd == "export":
        if args.round2:
            export_round2(args.entities, args.france_per_class, args.negatives, args.seed)
        else:
            export(args.entities, args.negatives, args.seed)
    elif args.cmd == "train":
        train(args.base_model, args.epochs, args.batch_size, args.lr, args.max_length, args.max_rows,
              train_file=args.train_file, out_dir=Path(args.out_model))
    elif args.cmd == "score":
        score(args.val_run, args.test_run, args.lo, args.batch_size, args.limit, model_dir=Path(args.model), tag=args.tag)
    else:
        blend(args.val_run, args.test_run, args.out_dir)


if __name__ == "__main__":
    main()
