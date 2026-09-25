"""Shared writer for output/candidate_pairs.tsv and output/matching_results.tsv.

Both files go through `write_id_list_tsv`, so they share one set of rules
(PROBLEM_STATEMENT.md "Output Format" + utils/validate_submission.py):

  * tab-separated, exact header, UTF-8, one row per test S1 entity in
    test_source1 order -- including entities with no candidates/matches,
    which get an empty list;
  * id lists comma-joined with no spaces or quoting, no duplicates;
  * "\n" line endings. Python on Windows would otherwise write "\r\n", and the
    validator/scorer strip only "\n" -- leaving a "\r" glued to the last id of
    every row, which silently turns that id into an unknown one.

`export_submission` also asserts matches are a subset of candidates: the
candidate file must be exactly what the model scored (PROBLEM_STATEMENT.md),
so a match outside it means a pipeline bug, not a formatting choice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"


def _joined_lists(source1_ids: np.ndarray, pairs: pd.DataFrame, id_col: str) -> list:
    """Comma-joined `id_col` per entry of `source1_ids` (in that order),
    highest score first; "" for entities with no pairs."""
    position = pd.Index(source1_ids).get_indexer(pairs["source1_entity_id"].to_numpy())
    if (position < 0).any():
        bad = pairs["source1_entity_id"].to_numpy()[position < 0][:5]
        raise ValueError(f"pairs reference S1 ids not in the required set, e.g. {list(bad)}")
    order = np.lexsort((-pairs["score"].to_numpy(), position))
    ids = pairs[id_col].to_numpy()[order]
    counts = np.bincount(position, minlength=len(source1_ids))
    bounds = np.concatenate([[0], np.cumsum(counts)])
    return [",".join(ids[bounds[i] : bounds[i + 1]]) for i in range(len(source1_ids))]


def write_id_list_tsv(path: Path, header: tuple, source1_ids: np.ndarray, pairs: pd.DataFrame, id_col: str) -> None:
    lists = _joined_lists(source1_ids, pairs, id_col)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        batch = 200_000
        for start in range(0, len(source1_ids), batch):
            f.write("".join(f"{s1}\t{ids}\n" for s1, ids in zip(source1_ids[start : start + batch], lists[start : start + batch])))
    tmp.replace(path)


def export_submission(
    out_dir: Path, source1_ids: np.ndarray, candidates: pd.DataFrame, matches: pd.DataFrame
) -> None:
    """candidates / matches: source1_entity_id, candidate_entity_id, score."""
    # 64-bit row hashes instead of concatenated key strings: ~69M test pairs.
    pair_cols = ["source1_entity_id", "candidate_entity_id"]
    cand_hash = pd.util.hash_pandas_object(candidates[pair_cols], index=False).to_numpy()
    match_hash = pd.util.hash_pandas_object(matches[pair_cols], index=False).to_numpy()
    assert np.isin(match_hash, cand_hash).all(), "a match is not among the scored candidates"
    assert len(np.unique(cand_hash)) == len(cand_hash), "duplicate candidate pair"

    out_dir.mkdir(parents=True, exist_ok=True)
    write_id_list_tsv(
        out_dir / CANDIDATE_FILE, ("source1_entity_id", "candidate_entity_ids"), source1_ids, candidates,
        "candidate_entity_id",
    )
    write_id_list_tsv(
        out_dir / MATCHING_FILE, ("source1_entity_id", "matched_entity_ids"), source1_ids, matches,
        "candidate_entity_id",
    )
