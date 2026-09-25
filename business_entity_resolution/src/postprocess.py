"""Scored candidate pairs -> final matches (plan.md Task 5).

Two guardrails, in order:

  1. Threshold: a pair is a match iff score >= the dev-tuned global threshold.
     An S1 entity with nothing above it gets an empty list (a predicted
     singleton).
  2. Conflict resolution: each S2/S3 id is kept for at most one S1 entity --
     its highest-scoring one. Justified by the ground truth itself: 0 of
     7,638,365 train pairs have an S2/S3 id claimed by two S1 entities. The S1
     side stays many-to-one (mean ~3.5 true matches per entity), so this is a
     sort + de-dupe by candidate id, not a 1:1 bipartite matching. Measured on
     dev_val: +0.0004 macro F0.5 (within dev_val only; on the full test set an
     id can collide across more entities, so the effect can only be larger).
"""

from __future__ import annotations

import pandas as pd


def select_matches(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """`pairs` has source1_entity_id, candidate_entity_id, score. Returns the
    subset of rows kept as matches (same columns)."""
    above = pairs[pairs["score"] >= threshold]
    # Highest score first; ties broken by S1 id so the result is deterministic.
    ranked = above.sort_values(
        ["candidate_entity_id", "score", "source1_entity_id"], ascending=[True, False, True], kind="mergesort"
    )
    return ranked.drop_duplicates("candidate_entity_id", keep="first").reset_index(drop=True)
