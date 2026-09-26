"""Unit tests for src/competition_features.py.

Run: python business_entity_resolution/tests/test_competition.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from competition_features import add_competition_features, competition  # noqa: E402


def test_competition_rank_margin_size():
    # candidate 0 has three S1 rivals, candidate 1 a tie, candidate 2 no rival
    codes = np.array([0, 0, 0, 1, 1, 2])
    values = np.array([0.9, 0.5, 0.7, 0.4, 0.4, 0.8])
    rank, margin, size = competition(codes, values)
    assert list(rank) == [0, 2, 1, 0, 1, 0]
    assert np.allclose(margin[:3], [0.2, -0.4, -0.2])  # winner: vs runner-up; others: vs winner
    assert list(margin[3:5]) == [0.0, 0.0]  # a tie is nobody's win
    assert np.isnan(margin[5])  # no rival
    assert list(size) == [3, 3, 3, 2, 2, 1]


def test_competition_nan_ranks_last():
    rank, margin, _ = competition(np.array([0, 0]), np.array([np.nan, 0.3]))
    assert list(rank) == [1, 0] and np.isnan(margin[0]) and np.isnan(margin[1])


def test_add_competition_features_spans_parts_and_is_rerunnable():
    rows = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-2"],
            "candidate_entity_id": ["S2-9", "S3-8", "S2-9", "S3-7"],
            "name_char3_idf_cos": [0.9, 0.2, 0.6, 0.5],
            "name_token_set_ratio": [1.0, 0.3, 0.8, 0.5],
            "emb_cos": [0.95, 0.4, 0.7, 0.6],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rows.iloc[:2].to_parquet(d / "part-00000.parquet", index=False)  # S2-9's rivals sit in two parts
        rows.iloc[2:].to_parquet(d / "part-00001.parquet", index=False)
        for _ in range(2):  # second run must replace, not duplicate, the columns
            add_competition_features(d)
        out = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("part-*.parquet"))], ignore_index=True)
    assert list(out.columns).count("comp_n_s1") == 1
    assert list(out["comp_n_s1"]) == [2, 1, 2, 1]
    assert list(out["comp_char3_rank"]) == [0, 0, 1, 0]
    assert np.isclose(out.loc[0, "comp_char3_margin"], 0.3) and np.isclose(out.loc[2, "comp_char3_margin"], -0.3)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
