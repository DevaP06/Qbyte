"""Unit tests for src/evaluate.py, src/threshold_tuning.py, and the entity
split in src/train_classifier.py.

Run: python business_entity_resolution/tests/test_evaluate.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import macro_f05, macro_f05_by_group, per_entity_f05  # noqa: E402
from threshold_tuning import tune_threshold  # noqa: E402
from train_classifier import ROLE_EARLY_STOP, ROLE_FIT, ROLE_VAL, split_entities  # noqa: E402


def test_problem_statement_worked_example():
    # Predicts [S2-00047, S2-00193, S3-00812]; truth [S2-00047, S3-00812]:
    # P = 2/3, R = 1 -> F_0.5 = 0.714 (PROBLEM_STATEMENT.md).
    f = per_entity_f05(np.array([2.0]), np.array([3.0]), np.array([2.0]))
    assert math.isclose(f[0], 0.7142857, rel_tol=1e-6), f


def test_singleton_rules():
    tp = np.array([0.0, 0.0, 0.0])
    n_pred = np.array([0.0, 1.0, 0.0])
    n_true = np.array([0.0, 0.0, 3.0])
    f = per_entity_f05(tp, n_pred, n_true)
    assert list(f) == [1.0, 0.0, 0.0]  # correct singleton, false merge on singleton, missed everything


def test_matches_textbook_formula():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n_true = rng.integers(1, 10)
        n_pred = rng.integers(1, 10)
        tp = rng.integers(1, min(n_true, n_pred) + 1)
        p, r = tp / n_pred, tp / n_true
        expected = 1.25 * p * r / (0.25 * p + r)
        got = per_entity_f05(np.array([tp], float), np.array([n_pred], float), np.array([n_true], float))[0]
        assert math.isclose(got, expected, rel_tol=1e-9)


def test_macro_average_includes_entities_without_pairs():
    # Entity 0: one correct match. Entity 1: singleton, no candidates at all.
    # Entity 2: has 2 true matches but blocking found none. Entity 3: not in eval set.
    entity = np.array([0, 3])
    label = np.array([1, 1])
    selected = np.array([True, True])
    n_true = np.array([1, 0, 2, 1])
    eval_mask = np.array([True, True, True, False])
    # (1.0 + 1.0 + 0.0) / 3 -- entity 2 counts as 0 even though it has no rows.
    assert math.isclose(macro_f05(entity, selected, label, n_true, eval_mask), 2 / 3)

    by = macro_f05_by_group(entity, selected, label, n_true, eval_mask, np.array(["US", "US", "India", "US"]))
    assert math.isclose(by["US"], 1.0) and math.isclose(by["India"], 0.0) and math.isclose(by["ALL"], 2 / 3)


def test_tune_threshold_prefers_precision():
    # Entity 0 true matches score 0.9, 0.8; a false candidate at 0.6.
    # Entity 1 is a singleton whose best (false) candidate scores 0.7.
    entity = np.array([0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.6, 0.7])
    label = np.array([1, 1, 0, 0])
    n_true = np.array([2, 0])
    result = tune_threshold(entity, scores, label, n_true, np.array([True, True]), thresholds=np.array([0.5, 0.65, 0.75, 0.85]))
    # 0.75 keeps both true matches and drops both false ones -> perfect 1.0.
    assert math.isclose(result.threshold, 0.75) and math.isclose(result.macro_f05, 1.0)


def test_split_entities_is_stratified_and_deterministic():
    countries = np.array(["US"] * 1000 + ["India"] * 600)
    roles = split_entities(countries, val_fraction=0.15, early_stop_fraction=0.05, seed=1)
    assert (roles == split_entities(countries, val_fraction=0.15, early_stop_fraction=0.05, seed=1)).all()
    for country, n in (("US", 1000), ("India", 600)):
        r = roles[countries == country]
        assert (r == ROLE_VAL).sum() == round(n * 0.15)
        assert (r == ROLE_EARLY_STOP).sum() == round(n * 0.05)
        assert (r == ROLE_FIT).sum() == n - round(n * 0.15) - round(n * 0.05)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
