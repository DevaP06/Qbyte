"""TSV I/O, chunked iteration, and the cleaning-stage cache helper.

This is the only module in the cleaning stage that depends on pandas/pyarrow
-- normalize.py and address_parser.py are stdlib-only and independently
testable. Kept deliberately minimal: just what this stage needs (chunked
read, a sample reader for smoke tests, one row-wise cleaning pass, and a
generic cache-or-build helper). The full pipeline CLI (`pipeline.py`,
per-stage subcommands, `--force`) belongs to a later stage once there is
more than one stage to orchestrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Optional

import pandas as pd

import config
from address_parser import parse_address
from normalize import normalize_country, normalize_text

SOURCE_COLUMNS = [config.ENTITY_ID_COL, config.NAME_COL, config.ADDRESS_COL, config.COUNTRY_COL]
GROUND_TRUTH_COLUMNS = [config.GT_SOURCE1_COL, config.GT_MATCHES_COL]


def read_source_chunks(
    path: Path, chunksize: int = config.DEFAULT_CHUNKSIZE
) -> Iterator[pd.DataFrame]:
    """Yield a source TSV in bounded-memory chunks.

    `dtype=str` avoids pandas' numeric/bool type inference on business
    names/addresses (e.g. a name that's all digits); missing cells still
    come through as float NaN, which `normalize.to_clean_str` handles.
    """
    return pd.read_csv(path, sep=config.TSV_SEP, dtype=str, chunksize=chunksize)


def read_sample(path: Path, n: int) -> pd.DataFrame:
    """Read just the first `n` data rows -- for smoke tests, not full runs."""
    return pd.read_csv(path, sep=config.TSV_SEP, dtype=str, nrows=n)


def read_ground_truth_chunks(
    path: Path, chunksize: int = config.DEFAULT_CHUNKSIZE
) -> Iterator[pd.DataFrame]:
    return pd.read_csv(path, sep=config.TSV_SEP, dtype=str, chunksize=chunksize)


def clean_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str = config.ENTITY_ID_COL,
    name_col: str = config.NAME_COL,
    address_col: str = config.ADDRESS_COL,
    country_col: str = config.COUNTRY_COL,
) -> pd.DataFrame:
    """Apply normalize_text/parse_address/normalize_country to one chunk.

    Transforms text representations only -- never drops rows, never touches
    `id_col`. The two assertions at the end are the data-integrity
    requirement (row count + entity IDs unchanged) enforced at this stage's
    exit boundary, not speculative validation.
    """
    name_results = df[name_col].map(normalize_text)
    address_results = df[address_col].map(parse_address)

    out = pd.DataFrame(
        {
            id_col: df[id_col].to_numpy(),
            country_col: df[country_col].map(normalize_country).to_numpy(),
            "name_original_normalized": [r.original_normalized for r in name_results],
            "name_ascii_folded": [r.ascii_folded for r in name_results],
            "name_tokens": [list(r.tokens) for r in name_results],
            "name_digit_tokens": [sorted(r.digit_tokens) for r in name_results],
            "address_present": [a.present for a in address_results],
            "address_normalized": [a.normalized for a in address_results],
            "address_head": [a.head for a in address_results],
            "address_middle": [list(a.middle) for a in address_results],
            "address_tail": [a.tail for a in address_results],
        }
    )

    assert len(out) == len(df), "cleaning must not change row count"
    assert (out[id_col].to_numpy() == df[id_col].to_numpy()).all(), (
        "cleaning must not change entity_id values or order"
    )
    return out


def clean_source_file_chunks(
    path: Path, chunksize: int = config.DEFAULT_CHUNKSIZE
) -> Iterator[pd.DataFrame]:
    """Chunked clean of a full source TSV -- reusable for the million-row files.

    Never materializes the whole file: reads one chunk, cleans it, yields it.
    Callers decide what to do with each cleaned chunk (write incrementally,
    aggregate stats, etc.).
    """
    for chunk in read_source_chunks(path, chunksize=chunksize):
        yield clean_dataframe(chunk)


def cache_or_build(
    cache_path: Path, build_fn: Callable[[], pd.DataFrame], force: bool = False
) -> pd.DataFrame:
    """Load `cache_path` (Parquet) if present, else build, persist, and return it.

    `force=True` recomputes and overwrites even if a cache exists. Never
    silently overwrites otherwise -- an existing cache is trusted as-is.
    """
    if not force and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = build_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df
