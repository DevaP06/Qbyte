"""Macro-F_0.5-optimal decision threshold (plan.md Task 4).

One global scalar threshold, never a per-country lookup table: a table keyed
by country string has no entry for France at test time. The search scores
every candidate threshold with evaluate.macro_f05 -- the exact metric,
singletons and candidate-less entities included -- rather than a proxy like
pair-level F_0.5 or logloss, since the per-entity macro average weighs a
false positive on a singleton (whole entity -> 0.0) far more heavily than a
pair-level metric would.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from evaluate import entity_counts, per_entity_f05


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    macro_f05: float
    curve: pd.DataFrame  # threshold -> macro_f05, for the write-up / plots


def candidate_thresholds(scores: np.ndarray, n_quantiles: int = 200) -> np.ndarray:
    """A fixed 0.01-step grid plus score quantiles, so the search resolves
    well wherever the scores actually concentrate (GBDT probabilities pile up
    near 0 and 1)."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 4)
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, n_quantiles)) if len(scores) else np.array([])
    return np.unique(np.concatenate([grid, quantiles]))


def tune_threshold(
    entity: np.ndarray,
    scores: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    eval_mask: np.ndarray,
    thresholds: np.ndarray | None = None,
    progress: Optional[Callable[[int, int, float], None]] = None,
) -> ThresholdResult:
    """Pick the threshold maximizing macro F_0.5 over `eval_mask` entities.
    A pair is predicted a match iff score >= threshold.

    `progress(done, total, best_so_far)` is called after each threshold."""
    # Re-index to just the evaluated entities (and their pairs): each
    # threshold then costs O(pairs + evaluated entities) instead of scanning
    # per-entity arrays for every train entity.
    eval_entities = np.flatnonzero(eval_mask)
    local = np.full(len(n_true), -1, dtype=np.int64)
    local[eval_entities] = np.arange(len(eval_entities))
    pair_local = local[entity]
    keep = pair_local >= 0
    entity, scores, label = pair_local[keep], scores[keep], label[keep]
    n_true = n_true[eval_entities]
    eval_mask = np.ones(len(eval_entities), dtype=bool)

    if thresholds is None:
        thresholds = candidate_thresholds(scores)
    n_entities = len(n_true)
    results = []
    for thr in thresholds:
        tp, n_pred = entity_counts(entity, scores >= thr, label, n_entities)
        results.append(per_entity_f05(tp, n_pred, n_true)[eval_mask].mean())
        if progress is not None:
            progress(len(results), len(thresholds), max(results))
    curve = pd.DataFrame({"threshold": thresholds, "macro_f05": results})
    best = int(np.argmax(curve["macro_f05"].to_numpy()))
    return ThresholdResult(
        threshold=float(curve["threshold"].iloc[best]),
        macro_f05=float(curve["macro_f05"].iloc[best]),
        curve=curve,
    )
