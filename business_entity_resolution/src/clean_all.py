"""Run the cleaning stage over the real train/test source files.

Reads each of the six source TSVs (train/test x source1/2/3) in
bounded-memory chunks, cleans each chunk via io_utils.clean_dataframe, and
streams the result straight to a Parquet file under
data/processed/<split>/<source>.parquet -- the full cleaned file is never
held in memory at once. Ground truth is untouched (it's ID pairs, nothing to
clean) and is not copied here.

Run:
    python business_entity_resolution/src/clean_all.py [--force]

--force recomputes and overwrites even if a cached output already exists;
otherwise an existing output for a given file is trusted as-is and skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

import config
from io_utils import clean_dataframe, cleaned_chunk_to_table, read_source_chunks


class ProgressBar:
    """Minimal stdlib real-time progress bar (in-place, carriage-return)."""

    def __init__(self, total: int, label: str, width: int = 30):
        self.total = total
        self.label = label
        self.width = width
        self.done = 0
        self.start = time.time()
        self._last_print = 0.0

    def update(self, n: int) -> None:
        self.done += n
        now = time.time()
        # Cap refresh rate so writing the bar doesn't itself become the
        # bottleneck; always show the final 100% update.
        if now - self._last_print < 0.2 and self.done < self.total:
            return
        self._last_print = now
        frac = min(self.done / self.total, 1.0) if self.total else 1.0
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else 0.0
        sys.stdout.write(
            f"\r  {self.label:14s} [{bar}] {frac * 100:5.1f}%  "
            f"{self.done:>9,}/{self.total:,}  {rate:8,.0f} rows/s  ETA {remaining:6.0f}s"
        )
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def count_data_rows(path: Path) -> int:
    """Fast line count minus the header, used only for the progress bar total."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def clean_file_to_parquet(src_path: Path, dest_path: Path, label: str, force: bool) -> int:
    if dest_path.exists() and not force:
        print(f"=== {label} === cached at {dest_path}, skipping (use --force to recompute)")
        return 0

    print(f"=== {label} === {src_path}")
    total = count_data_rows(src_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".parquet.tmp")

    bar = ProgressBar(total, label)
    writer = None
    rows_written = 0
    try:
        for chunk in read_source_chunks(src_path, chunksize=config.DEFAULT_CHUNKSIZE):
            cleaned = clean_dataframe(chunk)
            table = cleaned_chunk_to_table(cleaned)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)
            rows_written += len(cleaned)
            bar.update(len(cleaned))
    finally:
        if writer is not None:
            writer.close()
    bar.close()

    tmp_path.replace(dest_path)  # atomic on the same filesystem
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cleaning stage over all real source files.")
    parser.add_argument("--force", action="store_true", help="Recompute even if a cached output exists.")
    args = parser.parse_args()

    jobs = [
        ("train", "source1", config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["source1"]),
        ("train", "source2", config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["source2"]),
        ("train", "source3", config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["source3"]),
        ("test", "source1", config.DATA_RAW_TEST_DIR / config.TEST_FILES["source1"]),
        ("test", "source2", config.DATA_RAW_TEST_DIR / config.TEST_FILES["source2"]),
        ("test", "source3", config.DATA_RAW_TEST_DIR / config.TEST_FILES["source3"]),
    ]

    overall_start = time.time()
    total_rows = 0
    for split, source, src_path in jobs:
        dest_path = config.DATA_PROCESSED_DIR / split / f"{source}.parquet"
        label = f"{split}/{source}"
        rows = clean_file_to_parquet(src_path, dest_path, label, args.force)
        total_rows += rows

    elapsed = time.time() - overall_start
    if total_rows and elapsed > 0:
        print(f"\nDone: {total_rows:,} rows cleaned in {elapsed:.0f}s ({total_rows / elapsed:,.0f} rows/s overall)")
    else:
        print("\nDone (nothing to recompute).")


if __name__ == "__main__":
    main()
