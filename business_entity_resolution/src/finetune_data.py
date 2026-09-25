"""Training data for fine-tuning the retrieval embedding model (plan.md Task
1+3 dense retrieval; run on SageMaker or any GPU via finetune_embeddings.py).

Writes data/processed/finetune/{train,eval}.parquet with columns
(anchor, positive, negative) as record texts, plus ids/country for tracing:

  anchor    an S1 record
  positive  one of its ground-truth matches
  negative  a HARD negative: a non-matching record from the same S1's token
            blocking candidates (high blocking score = lexically close, e.g.
            "rosny culture sarl" for "rosny amis sarl") -- exactly the
            confusions retrieval has to rank below true matches.

Leakage guard: only `fit` entities of train_classifier.split_entities (same
seed) go into `train`, and early-stop entities into `eval`; dev_val entities
are never exported, so dev_val F0.5 stays an honest estimate after the
fine-tuned model is plugged into blocking and features.

Sampling: every ground-truth pair that token blocking MISSED (the recall gap
this model exists to close -- 27% of non-Latin India matches) is kept, then
the rest is filled with a uniform sample of the other pairs up to
--max-pairs. Rows are shuffled; the training script additionally uses a
no-duplicates batch sampler so two positives of one anchor never land in the
same batch as each other's in-batch negatives.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from io_utils import read_cleaned_table
from labels import PairLabeler, entity_truth, ground_truth_pairs, load_ground_truth
from train_classifier import ROLE_EARLY_STOP, ROLE_FIT, split_entities

FINETUNE_DIR = config.DATA_PROCESSED_DIR / "finetune"
HARD_NEGATIVES_PER_ENTITY = 5


def record_texts(split: str) -> pd.Series:
    """entity_id -> "name, address" text for every S1/S2/S3 record of a split.
    Built from the Unicode-preserving normalized fields, so non-Latin names
    reach the multilingual encoder intact. Retrieval must embed records with
    this same function."""
    parts = []
    for source in ("source1", "source2", "source3"):
        t = read_cleaned_table(split, source, columns=["entity_id", "name_original_normalized", "address_normalized"])
        name = pc.fill_null(t.column("name_original_normalized"), "")
        addr = pc.fill_null(t.column("address_normalized"), "")
        text = pc.if_else(pc.equal(addr, ""), name, pc.binary_join_element_wise(name, addr, ", "))
        parts.append(pd.Series(text.to_numpy(zero_copy_only=False), index=t.column("entity_id").to_numpy()))
    return pd.concat(parts)


def build_pairs(max_pairs: int, eval_pairs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    gt = load_ground_truth()
    truth = entity_truth(gt)
    roles = pd.Series(split_entities(truth["country"].to_numpy()), index=truth["source1_entity_id"].to_numpy())
    country = truth.set_index("source1_entity_id")["country"]

    pairs = ground_truth_pairs(gt)
    pairs["role"] = roles.reindex(pairs["source1_entity_id"]).to_numpy()
    pairs = pairs[pairs["role"].isin([ROLE_FIT, ROLE_EARLY_STOP])].reset_index(drop=True)

    # Streamed in record batches: the full 88M-row candidate table as pandas
    # objects would need 10GB+, and this often runs next to a training job.
    print("scanning train blocking candidates ...", flush=True)
    labeler = PairLabeler(pairs)
    wanted = pa.array(pairs["source1_entity_id"].unique())
    found_keys, neg_parts = [], []
    reader = pq.ParquetFile(config.CANDIDATES_DIR / "train" / "token_blocking.parquet")
    for batch in reader.iter_batches(batch_size=5_000_000, columns=["source1_entity_id", "candidate_entity_id", "score"]):
        batch = batch.filter(pc.is_in(batch.column("source1_entity_id"), value_set=wanted))
        chunk = batch.to_pandas()
        chunk["label"] = labeler.label(chunk["source1_entity_id"].to_numpy(), chunk["candidate_entity_id"].to_numpy())
        pos = chunk[chunk["label"] == 1]
        found_keys.append((pos["source1_entity_id"] + "|" + pos["candidate_entity_id"]).to_numpy())
        # Top-N per S1 within the batch; a group split across two batches is
        # re-ranked below, and the top-N of a union is within the union of
        # per-batch top-Ns, so this stays exact.
        neg = chunk[chunk["label"] == 0].sort_values(["source1_entity_id", "score"], ascending=[True, False])
        neg_parts.append(neg.groupby("source1_entity_id").head(HARD_NEGATIVES_PER_ENTITY))

    found = pd.Index(np.concatenate(found_keys))
    pairs["blocking_found"] = found.get_indexer(pairs["source1_entity_id"] + "|" + pairs["matched_id"]) >= 0

    neg = pd.concat(neg_parts).sort_values(["source1_entity_id", "score"], ascending=[True, False])
    neg = neg.groupby("source1_entity_id").head(HARD_NEGATIVES_PER_ENTITY)
    neg_lists = neg.groupby("source1_entity_id")["candidate_entity_id"].agg(list)
    del neg_parts, neg

    def pick_negative(s1_ids: np.ndarray) -> np.ndarray:
        lists = neg_lists.reindex(s1_ids)
        return np.array([rng.choice(l) if isinstance(l, list) and l else None for l in lists], dtype=object)

    def sample(frame: pd.DataFrame, n: int, keep_missed: bool) -> pd.DataFrame:
        if keep_missed:
            missed = frame[~frame["blocking_found"]]
            rest = frame[frame["blocking_found"]]
            n_rest = max(n - len(missed), 0)
            rest = rest.sample(n=min(n_rest, len(rest)), random_state=seed)
            frame = pd.concat([missed, rest])
        else:
            frame = frame.sample(n=min(n, len(frame)), random_state=seed)
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train = sample(pairs[pairs["role"] == ROLE_FIT], max_pairs, keep_missed=True)
    evalp = sample(pairs[pairs["role"] == ROLE_EARLY_STOP].drop_duplicates("source1_entity_id"), eval_pairs, keep_missed=False)

    print("attaching record texts ...", flush=True)
    texts = record_texts("train")
    out = []
    for frame in (train, evalp):
        negative_id = pick_negative(frame["source1_entity_id"].to_numpy())
        frame = frame.assign(negative_id=negative_id).dropna(subset=["negative_id"])
        out.append(
            pd.DataFrame(
                {
                    "anchor": texts.reindex(frame["source1_entity_id"]).to_numpy(),
                    "positive": texts.reindex(frame["matched_id"]).to_numpy(),
                    "negative": texts.reindex(frame["negative_id"]).to_numpy(),
                    "anchor_id": frame["source1_entity_id"].to_numpy(),
                    "positive_id": frame["matched_id"].to_numpy(),
                    "negative_id": frame["negative_id"].to_numpy(),
                    "country": country.reindex(frame["source1_entity_id"]).to_numpy(),
                    "blocking_found": frame["blocking_found"].to_numpy(),
                }
            )
        )
    return out[0], out[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export (anchor, positive, hard negative) fine-tuning data.")
    parser.add_argument("--max-pairs", type=int, default=1_500_000, help="Training rows (all blocking-missed pairs kept).")
    parser.add_argument("--eval-pairs", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    args = parser.parse_args()

    train, evalp = build_pairs(args.max_pairs, args.eval_pairs, args.seed)
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(FINETUNE_DIR / "train.parquet", index=False)
    evalp.to_parquet(FINETUNE_DIR / "eval.parquet", index=False)

    pd.set_option("display.width", 200)
    for name, frame in (("train", train), ("eval", evalp)):
        print(f"\n{name}: {len(frame):,} rows")
        print(frame.groupby("country").agg(rows=("anchor", "size"), blocking_missed=("blocking_found", lambda s: int((~s).sum()))))
    print("\nexample rows:")
    print(train[["anchor", "positive", "negative"]].head(3).to_string())
    print(f"\nwritten to {FINETUNE_DIR} -- upload this folder to S3 for the SageMaker job")


if __name__ == "__main__":
    main()
