"""Unit tests for src/labels.py.

Run: python business_entity_resolution/tests/test_labels.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from labels import (  # noqa: E402
    PairLabeler,
    entity_truth,
    ground_truth_pairs,
    iter_labeled_feature_parts,
    load_ground_truth,
)

GT_TSV = (
    "source1_entity_id\tmatched_entity_ids\n"
    "S1-1\tS2-10,S3-11\n"
    "S1-2\t\n"  # singleton
    "S1-3\tS3-30\n"
)


def _load_gt() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gt.tsv"
        path.write_text(GT_TSV, encoding="utf-8")
        return load_ground_truth(path)


def test_ground_truth_pairs_explodes_and_skips_singletons():
    pairs = ground_truth_pairs(_load_gt())
    assert list(map(tuple, pairs.to_numpy())) == [("S1-1", "S2-10"), ("S1-1", "S3-11"), ("S1-3", "S3-30")]


def test_pair_labeler():
    labeler = PairLabeler(ground_truth_pairs(_load_gt()))
    s1 = np.array(["S1-1", "S1-1", "S1-1", "S1-2", "S1-3", "S1-3"], dtype=object)
    cand = np.array(["S2-10", "S3-11", "S3-30", "S2-10", "S3-30", "S2-10"], dtype=object)
    # S3-30 is a true match of S1-3, not of S1-1: labels are per pair, not per candidate id.
    assert list(labeler.label(s1, cand)) == [1, 1, 0, 0, 1, 0]
    assert labeler.label(np.array([], dtype=object), np.array([], dtype=object)).shape == (0,)


def _with_processed_dir(fn):
    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp)
        (processed / "train").mkdir()
        pd.DataFrame(
            {"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US", "US", "France"]}
        ).to_parquet(processed / "train" / "source1.parquet", index=False)
        original = config.DATA_PROCESSED_DIR
        config.DATA_PROCESSED_DIR = processed
        try:
            return fn(processed)
        finally:
            config.DATA_PROCESSED_DIR = original


def test_entity_truth_counts_singletons_as_zero():
    truth = _with_processed_dir(lambda _: entity_truth(_load_gt())).set_index("source1_entity_id")
    assert truth.loc["S1-1", "n_true"] == 2
    assert truth.loc["S1-2", "n_true"] == 0
    assert truth.loc["S1-3", "n_true"] == 1
    assert truth.loc["S1-3", "country"] == "France"


def test_iter_labeled_feature_parts():
    labeler = PairLabeler(ground_truth_pairs(_load_gt()))

    def run(processed: Path):
        features_dir = processed / "features" / "train"
        features_dir.mkdir(parents=True)
        pd.DataFrame(
            {"source1_entity_id": ["S1-1", "S1-1"], "candidate_entity_id": ["S2-10", "S2-99"], "x": [0.9, 0.1]}
        ).to_parquet(features_dir / "part-00000.parquet", index=False)
        pd.DataFrame(
            {"source1_entity_id": ["S1-3"], "candidate_entity_id": ["S3-30"], "x": [0.8]}
        ).to_parquet(features_dir / "part-00001.parquet", index=False)
        return pd.concat(iter_labeled_feature_parts(features_dir, labeler), ignore_index=True)

    labeled = _with_processed_dir(run)
    assert list(labeled["label"]) == [1, 0, 1]
    assert list(labeled["x"]) == [0.9, 0.1, 0.8]  # features passed through untouched


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
