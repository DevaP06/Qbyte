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
from io_utils import read_cleaned_table
from labels import PairLabeler, entity_truth, ground_truth_pairs, load_ground_truth
from train_classifier import ROLE_EARLY_STOP, ROLE_FIT, ROLE_VAL, Progress, _stage, split_entities

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


_LEGAL_FR = frozenset("sarl sas sasu eurl sa sci snc ei selarl scp scm sca gie cie".split())


def _street_component(address: str) -> tuple[int, str, str]:
    """(index, component, house number) of the first comma component with a
    digit, or (-1, "", "")."""
    for i, comp in enumerate(address.split(",")):
        for tok in comp.split():
            if tok.isdigit():
                return i, comp.strip(), tok
    return -1, "", ""


def synthetic_french_negatives(pos: pd.DataFrame, texts_name: pd.Series, texts_addr: pd.Series, n: int, seed: int) -> pd.DataFrame:
    """Hard negatives built from confident French matches, mimicking the two
    distractor patterns France fails on (STATUS.md):
      word swap   - the candidate's business word replaced by another French
                    business word ("lutins comite sas" -> "lutins institut sas")
      street swap - same house number, another street of the same city+region
    Uses only test text (no external data). Returns S1 text / candidate text pairs, label 0."""
    rng = np.random.default_rng(seed)
    s1_ids, cand_ids = pos["source1_entity_id"].to_numpy(), pos["candidate_entity_id"].to_numpy()
    cand_name = texts_name.reindex(cand_ids).fillna("").to_numpy()
    cand_addr = texts_addr.reindex(cand_ids).fillna("").to_numpy()
    s1_text = (texts_name.reindex(s1_ids).fillna("") + ", " + texts_addr.reindex(s1_ids).fillna("")).str.strip(", ").to_numpy()

    # business-word vocabulary: frequent French S1 name tokens that are neither legal forms nor place words
    s1_names = texts_name.reindex(np.unique(s1_ids)).fillna("")
    place = set(t for a in texts_addr.reindex(np.unique(s1_ids)).fillna("") for t in a.replace(",", " ").split())
    counts = pd.Series([t for nm in s1_names for t in nm.split()]).value_counts()
    vocab = [t for t, c in counts.items() if c >= 200 and t.isalpha() and len(t) > 2 and t not in _LEGAL_FR and t not in place]
    vocab_set = set(vocab)

    # street donors grouped by the S1's non-street address components (city + region)
    city_key = [",".join(sorted(c.strip() for c in a.split(",") if not any(ch.isdigit() for ch in c))) for a in cand_addr]
    donors: dict = {}
    for key, a in zip(city_key, cand_addr):
        _, comp, _ = _street_component(a)
        if comp:
            donors.setdefault(key, []).append(comp)

    rows = []
    for i in rng.permutation(len(pos)):
        if len(rows) >= n:
            break
        if rng.random() < 0.8:  # word swap (tried first more often: it only applies when a business word is present)
            toks = cand_name[i].split()
            spots = [j for j, t in enumerate(toks) if t in vocab_set]
            if not spots:
                continue
            j = spots[rng.integers(len(spots))]
            new = vocab[rng.integers(len(vocab))]
            if new == toks[j]:
                continue
            toks[j] = new
            cand = ", ".join(x for x in (" ".join(toks), cand_addr[i]) if x)
            kind = "word_swap"
        else:  # street swap, same number
            idx, comp, number = _street_component(cand_addr[i])
            pool = donors.get(city_key[i], [])
            if idx < 0 or len(pool) < 2:
                continue
            donor = pool[rng.integers(len(pool))]
            _, _, d_num = _street_component(donor)
            new_comp = donor.replace(d_num, number, 1) if d_num else donor
            if new_comp.replace(number, "").strip() == comp.replace(number, "").strip():
                continue
            parts = cand_addr[i].split(",")
            parts[idx] = " " + new_comp if idx else new_comp
            cand = ", ".join(x for x in (cand_name[i], ",".join(parts).strip()) if x)
            kind = "street_swap"
        rows.append((s1_ids[i], f"synthetic:{cand_ids[i]}", s1_text[i], cand, 0, kind))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "label", "kind"])


_ABBREV = [("avenue", "av"), ("boulevard", "bd"), ("rue", "r"), ("place", "pl"), ("chemin", "ch"),
           ("impasse", "imp"), ("allée", "all"), ("allee", "all"), ("route", "rte"), ("quai", "qu")]


def _fold_accents(text: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def synthetic_french_positives(pos: pd.DataFrame, texts: pd.Series, n: int, seed: int) -> pd.DataFrame:
    """Noisy copies of confident French matches (label stays 1): legal form
    stripped, accents folded, street type abbreviated/expanded, articles dropped,
    'bis' toggled, or a one-character typo -- the variations a true French
    match shows, so the CE learns they do NOT signal a different business."""
    rng = np.random.default_rng(seed + 7)
    rows = []
    sample = pos.sample(min(n, len(pos)), random_state=seed)
    for s1, cand in zip(sample["source1_entity_id"], sample["candidate_entity_id"]):
        a, b = texts.get(s1, ""), texts.get(cand, "")
        if not a or not b:
            continue
        for _ in range(int(rng.integers(1, 3))):
            op = int(rng.integers(6))
            if op == 0:
                b = " ".join(t for t in b.split(" ") if t.strip(",") not in _LEGAL_FR)
            elif op == 1:
                b = _fold_accents(b)
            elif op == 2:
                full, short = _ABBREV[int(rng.integers(len(_ABBREV)))]
                b = b.replace(f" {full} ", f" {short} ", 1) if f" {full} " in b else b.replace(f" {short} ", f" {full} ", 1)
            elif op == 3:
                for art in (" de la ", " du ", " des ", " de l'", " d'"):
                    b = b.replace(art, " ", 1)
            elif op == 4:
                b = b.replace(" bis ", " ", 1) if " bis " in b else b
            else:
                toks = b.split(" ")
                idx = [i for i, t in enumerate(toks) if t.isalpha() and len(t) > 3]
                if idx:
                    i = idx[int(rng.integers(len(idx)))]
                    j = int(rng.integers(len(toks[i]) - 1))
                    t = toks[i]
                    toks[i] = t[:j] + t[j + 1] + t[j] + t[j + 2:]
                    b = " ".join(toks)
        rows.append((s1, f"synthetic_pos:{cand}", a, " ".join(b.split()), 1))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "label"])


def s1_s1_negatives(n: int, seed: int) -> pd.DataFrame:
    """Free, CORRECT negatives for France: Source 1 is deduplicated, so any two
    distinct S1 records are different entities. Mine French S1 pairs in the same
    city+region that share the first name word ("lille ecole sas" vs "lille
    ecole club") or the street component (a different business at the same
    street) -- real versions of the distractors France fails on."""
    rng = np.random.default_rng(seed + 11)
    t = read_cleaned_table("test", "source1", columns=["entity_id", "country", "name_original_normalized", "address_normalized"]).to_pandas()
    t = t[t["country"] == "France"].fillna("")
    t["text"] = (t["name_original_normalized"] + ", " + t["address_normalized"]).str.strip(", ")
    t["city"] = [",".join(sorted(c.strip() for c in a.split(",") if c.strip() and not any(ch.isdigit() for ch in c))) for a in t["address_normalized"]]
    t["head"] = t["name_original_normalized"].str.split(" ").str[0]
    t["street"] = [_street_component(a)[1] for a in t["address_normalized"]]
    pairs = []
    per_kind = n // 2
    for key_cols in (["city", "head"], ["city", "street"]):
        groups = [g.index.to_numpy() for _, g in t[t[key_cols[1]] != ""].groupby(key_cols) if len(g) > 1]
        rng.shuffle(groups)
        taken = 0
        for idx in groups:
            if taken >= per_kind:
                break
            i, j = rng.choice(idx, size=2, replace=False)
            if t.at[i, "text"] != t.at[j, "text"]:
                pairs.append((t.at[i, "entity_id"], t.at[j, "entity_id"], t.at[i, "text"], t.at[j, "text"], 0))
                taken += 1
    return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "label"])


def _assert_no_dev_val(frame: pd.DataFrame, truth: pd.DataFrame, roles: np.ndarray) -> None:
    """Global rule: no dev_val entity may appear in any CE training set."""
    val_ids = set(truth["source1_entity_id"].to_numpy()[roles == ROLE_VAL])
    leaked = set(frame["source1_entity_id"]) & val_ids
    assert not leaked, f"dev_val leakage: {len(leaked)} dev_val entities in CE training data"


def export_round3(n_entities: int, france_per_class: int, n_synthetic: int, n_synth_pos: int, n_s1_neg: int,
                  negatives_per_entity: int, seed: int) -> None:
    """Round-3 data (all from fit entities or unlabeled test; dev_val asserted absent):
      replay      fresh India/US fit entities round 1/2 never saw (no forgetting)
      agreement   France pairs where GBDT and the ROUND-2 CE agree (>= 0.98 match;
                  CE <= 0.02 with GBDT < 0.5 non-match)
      synth neg   French business-word swaps / same number, other street
      synth pos   noisy copies of confident French matches (legal form, accents,
                  abbreviations, articles, bis, typo)
      S1-S1 neg   distinct French S1 records sharing name head or street (dedup => different)"""
    rng = np.random.default_rng(seed + 2)
    gt = load_ground_truth()
    truth = entity_truth(gt)
    roles = split_entities(truth["country"].to_numpy())
    used = set()
    for f in ("train.parquet", "train_r2.parquet"):
        used |= set(pd.read_parquet(CE_DIR / f, columns=["source1_entity_id"])["source1_entity_id"])
    fresh = np.array([i for i in truth["source1_entity_id"].to_numpy()[roles == ROLE_FIT] if i not in used])
    replay = _labelled_pairs(set(rng.choice(fresh, size=min(n_entities, len(fresh)), replace=False)), gt, negatives_per_entity, seed)
    _assert_no_dev_val(replay, truth, roles)

    _stage("France agreement pairs (GBDT + round-2 CE)")
    ts = pd.read_parquet(CE_DIR / "test_scores_r2.parquet")
    fr = ts[ts["country"] == "France"]
    pos = fr[(fr["score"] >= 0.98) & (fr["ce"] >= 0.98)]
    pos = pos.sort_values("ce", ascending=False).drop_duplicates("candidate_entity_id")  # one S1 per French record
    neg = fr[(fr["ce"] <= 0.02) & (fr["score"] < 0.5)]
    pos_s = pos.sample(min(france_per_class, len(pos)), random_state=seed).assign(label=1)
    neg_s = neg.sample(min(france_per_class, len(neg)), random_state=seed).assign(label=0)
    texts = record_texts("test")
    agree = pd.concat([pos_s, neg_s])
    agree = agree.assign(
        text_a=texts.reindex(agree["source1_entity_id"]).to_numpy(),
        text_b=texts.reindex(agree["candidate_entity_id"]).to_numpy(),
    )[["source1_entity_id", "candidate_entity_id", "text_a", "text_b", "label"]]

    _stage("synthetic French negatives + positives, and S1-S1 negatives")
    cleaned = pd.concat([read_cleaned_table("test", s, columns=["entity_id", "name_original_normalized", "address_normalized"]).to_pandas()
                         for s in ("source1", "source2", "source3")]).set_index("entity_id")
    synth = synthetic_french_negatives(pos, cleaned["name_original_normalized"], cleaned["address_normalized"], n_synthetic, seed)
    synth_pos = synthetic_french_positives(pos, texts, n_synth_pos, seed)
    s1neg = s1_s1_negatives(n_s1_neg, seed)

    out = pd.concat([replay, agree, synth.drop(columns="kind"), synth_pos, s1neg]).sample(frac=1.0, random_state=seed)
    out.to_parquet(CE_DIR / "train_r3.parquet", index=False)
    print(f"  round-3 train {len(out):,} pairs: replay {len(replay):,} | France agreement {len(agree):,} "
          f"({len(pos_s):,} match / {len(neg_s):,} non-match) | synthetic neg {len(synth):,} {synth['kind'].value_counts().to_dict()} "
          f"| synthetic pos {len(synth_pos):,} | S1-S1 neg {len(s1neg):,} | positive share {out['label'].mean() * 100:.1f}%", flush=True)
    for title, frame in (("synthetic NEG", synth.head(2)), ("synthetic POS", synth_pos.head(2)), ("S1-S1 NEG", s1neg.head(2))):
        for _, r in frame.iterrows():
            print(f"    [{title}] {r.text_a}\n       vs {r.text_b}", flush=True)


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
    e.add_argument("--round3", action="store_true", help="Replay + France agreement (round-2 CE) + synthetic negatives -> train_r3.parquet.")
    e.add_argument("--synthetic", type=int, default=150_000, help="Round 3: synthetic French hard negatives.")
    e.add_argument("--synthetic-pos", type=int, default=100_000, help="Round 3: synthetic French positives.")
    e.add_argument("--s1-negatives", type=int, default=150_000, help="Round 3: French S1-S1 negatives.")
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
        if args.round3:
            export_round3(args.entities, args.france_per_class, args.synthetic, args.synthetic_pos, args.s1_negatives,
                          args.negatives, args.seed)
        elif args.round2:
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
