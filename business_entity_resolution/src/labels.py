"""Ground-truth labels for candidate pairs (plan.md build-order step 3).

Two things downstream stages need from train_ground_truth.tsv:

  * a pair label (1 = the candidate is a true match of that S1 entity) for
    every row of the features dataset -- the classifier's target;
  * per-S1-entity truth (`n_true`, including 0 for singletons and entities
    blocking never gave a candidate) -- the recall denominator and singleton
    rule of the macro F_0.5 metric. Computing F_0.5 only over entities that
    have candidates would silently overstate the score.

Labels are computed on the fly rather than cached as a separate Parquet
dataset: one hash lookup over "s1|candidate" keys (measured ~0.26s per 1M
pairs, ~25s for all of train) is cheap enough that a cache would only add a
way for labels to go stale against re-run features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from io_utils import read_cleaned_table

_KEY_SEP = "|"


def load_ground_truth(path: Optional[Path] = None) -> pd.DataFrame:
    """One row per train S1 entity: (source1_entity_id, matched_entity_ids),
    with singletons' empty lists read as ""."""
    path = path or config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["ground_truth"]
    gt = pd.read_csv(path, sep=config.TSV_SEP, dtype=str, keep_default_na=False)
    return gt.rename(columns={config.GT_SOURCE1_COL: "source1_entity_id", config.GT_MATCHES_COL: "matched_entity_ids"})


def ground_truth_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Explode to one (source1_entity_id, matched_id) row per true match."""
    matched = gt[gt["matched_entity_ids"].str.strip() != ""]
    pairs = matched.assign(matched_id=matched["matched_entity_ids"].str.split(","))
    pairs = pairs.explode("matched_id")[["source1_entity_id", "matched_id"]]
    pairs["matched_id"] = pairs["matched_id"].str.strip()
    return pairs.reset_index(drop=True)


def _pair_keys(source1_ids, candidate_ids) -> np.ndarray:
    return pc.binary_join_element_wise(
        pa.array(source1_ids, type=pa.string()), pa.array(candidate_ids, type=pa.string()), _KEY_SEP
    ).to_numpy(zero_copy_only=False)


class PairLabeler:
    """Label arbitrary (S1, candidate) id pairs against the ground truth.

    The positive-key Index's hash table is built on first lookup and reused
    by every later call, so labeling a features dataset part by part costs
    one build (~4s for 7.6M train pairs) plus a cheap probe per part.
    """

    def __init__(self, pairs: pd.DataFrame):
        self._positives = pd.Index(_pair_keys(pairs["source1_entity_id"].to_numpy(), pairs["matched_id"].to_numpy()))

    def label(self, source1_ids, candidate_ids) -> np.ndarray:
        return (self._positives.get_indexer(_pair_keys(source1_ids, candidate_ids)) >= 0).astype(np.int8)


def entity_truth(gt: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    """Every S1 entity of `split` with its country and number of true
    matches (0 = singleton). Asserts GT covers exactly the split's S1 ids."""
    s1 = read_cleaned_table(split, "source1", columns=["entity_id", "country"]).to_pandas()
    matched = gt["matched_entity_ids"].str.strip()
    n_true = np.where(matched == "", 0, matched.str.count(",") + 1)
    truth = pd.DataFrame({"source1_entity_id": gt["source1_entity_id"].to_numpy(), "n_true": n_true.astype(np.int16)})
    out = s1.rename(columns={"entity_id": "source1_entity_id"}).merge(truth, on="source1_entity_id", how="left")
    assert len(out) == len(s1) == len(gt), "ground truth must have exactly one row per S1 entity"
    assert out["n_true"].notna().all(), "S1 entity missing from ground truth"
    return out


def iter_labeled_feature_parts(
    features_dir: Path, labeler: PairLabeler, columns: Optional[list] = None
) -> Iterator[pd.DataFrame]:
    """Yield each features part (see features.py) with a `label` column added."""
    for part in sorted(features_dir.glob("part-*.parquet")):
        df = pq.read_table(part, columns=columns).to_pandas()
        df["label"] = labeler.label(df["source1_entity_id"].to_numpy(), df["candidate_entity_id"].to_numpy())
        yield df


def main() -> None:
    """Label-coverage report for a features dataset: how many true pairs the
    candidate set (and so any classifier on it) can possibly recover."""
    parser = argparse.ArgumentParser(description="Report label coverage of a train features dataset.")
    parser.add_argument("--features", default="train", help="Directory name under data/processed/features/.")
    args = parser.parse_args()

    gt = load_ground_truth()
    pairs = ground_truth_pairs(gt)
    truth = entity_truth(gt).set_index("source1_entity_id")
    labeler = PairLabeler(pairs)

    per_entity = []
    for df in iter_labeled_feature_parts(
        config.FEATURES_DIR / args.features, labeler, columns=["source1_entity_id", "candidate_entity_id", "country"]
    ):
        per_entity.append(df.groupby("source1_entity_id").agg(n_cand=("label", "size"), n_found=("label", "sum")))
    found = pd.concat(per_entity)
    covered = truth.loc[found.index]  # sampled datasets only cover some entities

    report = covered.join(found)
    report["all_found"] = report["n_found"] == report["n_true"]
    by_country = report.groupby("country").agg(
        entities=("n_true", "size"),
        singletons=("n_true", lambda s: int((s == 0).sum())),
        true_pairs=("n_true", "sum"),
        found_pairs=("n_found", "sum"),
        candidates=("n_cand", "sum"),
        entities_all_found=("all_found", "mean"),
    )
    by_country["pair_recall"] = by_country["found_pairs"] / by_country["true_pairs"]
    by_country["positive_rate"] = by_country["found_pairs"] / by_country["candidates"]
    n_no_candidates = len(truth) - len(found) if args.features == "train" else None

    pd.set_option("display.width", 200)
    print(f"=== label coverage: features/{args.features} ===")
    print(by_country.to_string(float_format=lambda x: f"{x:.4f}"))
    if n_no_candidates is not None:
        print(f"  S1 entities with no candidates (not in features): {n_no_candidates:,}")


if __name__ == "__main__":
    main()
