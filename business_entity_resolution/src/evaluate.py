"""Local macro F_0.5, exactly as PROBLEM_STATEMENT.md scores it.

Per S1 entity, with TP/FP/FN counted over its predicted vs. true match sets:

    F_0.5 = 1.25*TP / (1.25*TP + 0.25*FN + FP)

which is the textbook (1+b^2)PR/(b^2 P + R) rewritten in counts, so it needs
no special-casing for P or R being undefined -- except the one case the
problem statement defines explicitly: predicted empty AND truly empty (a
correctly-identified singleton) scores 1.0, where the formula is 0/0. Every
other edge case falls out of the formula on its own:

    singleton, predicted anything     -> FP > 0, TP = 0  -> 0.0
    has matches, predicted nothing    -> FN > 0, TP = 0  -> 0.0

The macro average runs over EVERY S1 entity of the evaluation set, including
entities blocking gave no candidates to (they predict empty by construction)
-- averaging only over entities present in the features would overstate the
score. Entities are therefore passed as integer codes into a full per-entity
truth table (labels.entity_truth), not discovered from the pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_entity_f05(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    fp = n_pred - tp
    fn = n_true - tp
    num = 1.25 * tp
    den = 1.25 * tp + 0.25 * fn + fp
    out = np.ones(len(tp), dtype=np.float64)  # 0/0: empty predicted, empty true
    np.divide(num, den, out=out, where=den > 0)
    return out


def entity_counts(
    entity: np.ndarray, selected: np.ndarray, label: np.ndarray, n_entities: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-entity (TP, number predicted) for the pairs where `selected`."""
    ent = entity[selected]
    tp = np.bincount(ent, weights=label[selected], minlength=n_entities)
    n_pred = np.bincount(ent, minlength=n_entities).astype(np.float64)
    return tp, n_pred


def macro_f05(
    entity: np.ndarray,
    selected: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    eval_mask: np.ndarray,
) -> float:
    """Macro F_0.5 over the entities where `eval_mask` is True.

    entity:    per-pair entity code (index into n_true / eval_mask)
    selected:  per-pair bool, the pairs predicted as matches
    label:     per-pair 0/1 ground truth
    n_true:    per-entity true match count (0 for singletons), for ALL entities
    eval_mask: per-entity bool, which entities this score averages over
    """
    tp, n_pred = entity_counts(entity, selected, label, len(n_true))
    return float(per_entity_f05(tp, n_pred, n_true)[eval_mask].mean())


def macro_f05_by_group(
    entity: np.ndarray,
    selected: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    eval_mask: np.ndarray,
    groups: np.ndarray,
) -> pd.Series:
    """Macro F_0.5 per value of a per-entity `groups` array (e.g. country),
    plus an "ALL" row."""
    tp, n_pred = entity_counts(entity, selected, label, len(n_true))
    f = per_entity_f05(tp, n_pred, n_true)
    s = pd.Series(f[eval_mask]).groupby(groups[eval_mask]).mean()
    s["ALL"] = f[eval_mask].mean()
    return s
