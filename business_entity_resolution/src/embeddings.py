"""Dense retrieval: embed every record with the (fine-tuned) multilingual
encoder and take each S1's nearest S2/S3 neighbours within its country
(plan.md Task 1+3, the secondary blocking layer).

Why: measured on a 5% train sample, the true matches token blocking misses
are mostly character-level and cross-script noise -- typos ("kd-fottcabre"),
merged/split words ("mubarakpolymers"), reordering, truncated addresses, and
Devanagari/Kannada names -- that no word-token key can retrieve. Keeping 300
token candidates instead of 40 recovers only +4.4 (India) / +1.6 (US) recall
points; name char-trigram retrieval adds +2.8-4.4 / +1.4-2.4 and nothing for
non-Latin names. An encoder over the whole "name, address" text targets all
of these at once.

Stages (each cached; --force recomputes):

  embed  data/processed/embeddings/<model>/<split>.f16 -- float16,
         L2-normalized, one row per record in record_texts() order
         (source1, source2, source3), with <split>.ids.npy alongside.
         Identical texts are encoded once.
  knn    data/processed/candidates/<split>/embedding_<model>.parquet --
         source1_entity_id, candidate_entity_id, emb_score (cosine), emb_rank.
         Exact search (a matmul, no ANN approximation) per country: the same
         "group by whatever string is in `country`" rule as token blocking,
         so France needs no special case. Scored in query batches sized so
         one similarity block stays under --block-bytes of GPU memory.
  recall (--recall, train only) recall of token@40, embedding@k, and their
         union on the dev_val entities (never used for fine-tuning), by
         country and script -- the go/no-go measurement for this layer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import config
from finetune_data import record_texts
from io_utils import read_cleaned_table
from labels import entity_truth, ground_truth_pairs, load_ground_truth
from train_classifier import ROLE_VAL, Progress, _stage, split_entities

EMBEDDINGS_DIR = config.DATA_PROCESSED_DIR / "embeddings"
DEFAULT_PREFIX = "query: "  # E5 convention; finetune_config.json overrides it


def model_key(model: str) -> str:
    """Filesystem-safe name for a model dir or hub id."""
    p = Path(model)
    return p.name if p.exists() else model.replace("/", "__")


def model_prefix(model: str) -> str:
    cfg = Path(model) / "finetune_config.json"
    return json.loads(cfg.read_text()).get("prefix", DEFAULT_PREFIX) if cfg.exists() else DEFAULT_PREFIX


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def record_table(split: str) -> pd.DataFrame:
    """entity_id, country, is_source1, text -- one row per record, in the
    same order as record_texts() (and therefore the embedding matrix)."""
    texts = record_texts(split)
    parts = [read_cleaned_table(split, s, columns=["entity_id", "country"]).to_pandas() for s in ("source1", "source2", "source3")]
    meta = pd.concat(parts, ignore_index=True)
    meta["is_source1"] = np.arange(len(meta)) < len(parts[0])
    assert (meta["entity_id"].to_numpy() == texts.index.to_numpy()).all(), "record order mismatch"
    meta["text"] = texts.to_numpy()
    return meta


# --------------------------------------------------------------------------
# Embed
# --------------------------------------------------------------------------


def embed_texts(texts: np.ndarray, encoder, out: np.ndarray, prefix: str, chunk: int = 500_000, batch_size: int = 512) -> None:
    """Encode `texts` into `out` (float16 rows, L2-normalized). Unique texts
    are encoded once. `encoder` is anything with SentenceTransformer's
    encode(list, batch_size=, normalize_embeddings=, convert_to_numpy=,
    show_progress_bar=) signature (a stub in tests)."""
    codes, uniques = pd.factorize(texts)
    bar = Progress(len(uniques), "embedding", "texts")
    uniq_emb = np.empty((len(uniques), out.shape[1]), dtype=np.float16)
    for s in range(0, len(uniques), chunk):
        batch = [prefix + t for t in uniques[s : s + chunk]]
        uniq_emb[s : s + len(batch)] = encoder.encode(
            batch, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float16)
        bar.update(s + len(batch))
    bar.close()
    out[:] = uniq_emb[codes]


def embed_split(split: str, model: str, force: bool = False, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(ids, float16 memmap) for every record of `split`; cached."""
    from sentence_transformers import SentenceTransformer

    base = EMBEDDINGS_DIR / model_key(model)
    emb_path, ids_path = base / f"{split}.f16", base / f"{split}.ids.npy"
    if emb_path.exists() and ids_path.exists() and not force:
        ids = np.load(ids_path, allow_pickle=True)
        dim = emb_path.stat().st_size // (2 * len(ids))
        return ids, np.memmap(emb_path, dtype=np.float16, mode="r", shape=(len(ids), dim))

    records = record_table(split)
    encoder = SentenceTransformer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        encoder.half()
    dim = encoder.get_sentence_embedding_dimension()
    base.mkdir(parents=True, exist_ok=True)
    tmp = emb_path.with_suffix(".f16.tmp")
    out = np.memmap(tmp, dtype=np.float16, mode="w+", shape=(len(records), dim))
    embed_texts(records["text"].to_numpy(), encoder, out, model_prefix(model), batch_size=batch_size)
    out.flush()
    del out
    tmp.replace(emb_path)
    np.save(ids_path, records["entity_id"].to_numpy(), allow_pickle=True)
    return embed_split(split, model)


# --------------------------------------------------------------------------
# kNN
# --------------------------------------------------------------------------


def knn_by_country(
    emb: np.ndarray,
    countries: np.ndarray,
    is_source1: np.ndarray,
    k: int,
    device: str = "cuda",
    block_bytes: float = 2e9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact top-k (cosine = dot product of normalized rows) from each
    source1 row to the non-source1 rows of the same country.
    Returns (query_row, target_row, score) as global row indices."""
    dtype = torch.float16 if device.startswith("cuda") else torch.float32  # CPU matmul has no fast fp16
    q_out, t_out, s_out = [], [], []
    for country in np.unique(countries[is_source1]):
        q_rows = np.flatnonzero(is_source1 & (countries == country))
        t_rows = np.flatnonzero(~is_source1 & (countries == country))
        if len(t_rows) == 0:
            continue
        kk = min(k, len(t_rows))
        T = torch.from_numpy(np.ascontiguousarray(emb[t_rows])).to(device=device, dtype=dtype)
        q_batch = max(1, int(block_bytes // (len(t_rows) * T.element_size())))
        bar = Progress(len(q_rows), f"knn {country}", "S1")
        for s in range(0, len(q_rows), q_batch):
            rows = q_rows[s : s + q_batch]
            Q = torch.from_numpy(np.ascontiguousarray(emb[rows])).to(device=device, dtype=dtype)
            scores, idx = (Q @ T.T).topk(kk, dim=1)
            q_out.append(np.repeat(rows, kk))
            t_out.append(t_rows[idx.cpu().numpy().ravel()])
            s_out.append(scores.float().cpu().numpy().ravel())
            bar.update(s + len(rows))
        bar.close()
        del T
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not q_out:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    return np.concatenate(q_out), np.concatenate(t_out), np.concatenate(s_out)


def knn_candidates(split: str, model: str, k: int, force: bool = False, block_bytes: float = 2e9) -> pd.DataFrame:
    dest = config.CANDIDATES_DIR / split / f"embedding_{model_key(model)}.parquet"
    if dest.exists() and not force:
        return pd.read_parquet(dest)
    ids, emb = embed_split(split, model)
    meta = record_table(split)[["entity_id", "country", "is_source1"]]
    assert (meta["entity_id"].to_numpy() == ids).all(), "embedding rows out of sync with records"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, t, s = knn_by_country(emb, meta["country"].to_numpy(), meta["is_source1"].to_numpy(), k, device, block_bytes)
    cands = pd.DataFrame({"source1_entity_id": ids[q], "candidate_entity_id": ids[t], "emb_score": s})
    cands["emb_rank"] = cands.groupby("source1_entity_id").cumcount().astype(np.int16)  # topk is score-desc
    dest.parent.mkdir(parents=True, exist_ok=True)
    cands.to_parquet(dest, index=False)
    return cands


# --------------------------------------------------------------------------
# Recall (go/no-go)
# --------------------------------------------------------------------------


def recall_report(emb_cands: pd.DataFrame, ks=(5, 10, 20, 50)) -> pd.DataFrame:
    """Recall on dev_val entities of token@40, embedding@k, and their union,
    by country and by script of the true match's name."""
    gt = load_ground_truth()
    truth = entity_truth(gt)
    roles = split_entities(truth["country"].to_numpy())
    val_ids = set(truth.loc[roles == ROLE_VAL, "source1_entity_id"])
    pairs = ground_truth_pairs(gt)
    pairs = pairs[pairs["source1_entity_id"].isin(val_ids)].copy()
    key = lambda a, b: a.astype(str) + "|" + b.astype(str)
    pair_key = key(pairs["source1_entity_id"], pairs["matched_id"])

    token = pq.read_table(
        config.CANDIDATES_DIR / "train" / "token_blocking.parquet",
        columns=["source1_entity_id", "candidate_entity_id"],
        filters=[("source1_entity_id", "in", list(val_ids))],
    ).to_pandas()
    pairs["token"] = pair_key.isin(set(key(token["source1_entity_id"], token["candidate_entity_id"]))).to_numpy()

    names = pd.concat(
        [read_cleaned_table("train", s, columns=["entity_id", "country", "name_original_normalized", "name_ascii_folded"]).to_pandas()
         for s in ("source2", "source3")]
    ).set_index("entity_id").reindex(pairs["matched_id"])
    pairs["country"] = names["country"].to_numpy()
    pairs["script"] = np.where(
        names["name_ascii_folded"].fillna("").str.len() < 0.5 * names["name_original_normalized"].fillna("").str.len(),
        "non-Latin", "Latin",
    )

    e = emb_cands[emb_cands["source1_entity_id"].isin(val_ids)]
    rows = []
    cols = {"token@40": pairs["token"]}
    for k in ks:
        hit = pair_key.isin(set(key(e.loc[e["emb_rank"] < k, "source1_entity_id"], e.loc[e["emb_rank"] < k, "candidate_entity_id"]))).to_numpy()
        cols[f"emb@{k}"] = hit
        cols[f"union@{k}"] = pairs["token"].to_numpy() | hit
    frame = pd.DataFrame(cols)
    frame["group"] = pairs["country"].to_numpy() + " " + pairs["script"].to_numpy()
    out = frame.groupby("group").mean() * 100
    all_row = frame.drop(columns="group").mean() * 100
    out.loc["ALL"] = all_row
    out.insert(0, "pairs", frame.groupby("group").size().reindex(out.index).fillna(len(frame)).astype(int))
    return out.round(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed records and retrieve nearest-neighbour candidates.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model", default=str(config.MODELS_DIR / "retriever"),
                        help="Fine-tuned model dir or a hub id (e.g. intfloat/multilingual-e5-small).")
    parser.add_argument("--k", type=int, default=50, help="Neighbours kept per S1 entity.")
    parser.add_argument("--batch-size", type=int, default=512, help="Encoding batch size.")
    parser.add_argument("--block-bytes", type=float, default=2e9, help="GPU memory per similarity block.")
    parser.add_argument("--recall", action="store_true", help="train: report dev_val recall vs token blocking.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    _stage(f"embedding {args.split} records with {args.model}")
    embed_split(args.split, args.model, force=args.force, batch_size=args.batch_size)
    _stage(f"nearest neighbours (k={args.k}) per country")
    cands = knn_candidates(args.split, args.model, args.k, force=args.force, block_bytes=args.block_bytes)
    print(f"  {len(cands):,} embedding candidate pairs", flush=True)

    if args.recall and args.split == "train":
        _stage("dev_val recall: token@40 vs embedding@k vs union")
        pd.set_option("display.width", 200)
        print(recall_report(cands).to_string())
    _stage(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
