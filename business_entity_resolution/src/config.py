"""Paths, column names, and shared constants for the cleaning stage.

No hyperparameters for later stages (blocking/embedding/model) live here yet
- only what normalize.py/address_parser.py/io_utils.py need. Extend this file
as later stages are implemented rather than pre-declaring their settings now.
"""

from pathlib import Path

# business_entity_resolution/src/config.py -> parents[1] is business_entity_resolution,
# parents[2] is the repo root.
SRC_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SRC_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent

RESOURCES_DIR = SRC_DIR / "resources"
LEGAL_SUFFIXES_PATH = RESOURCES_DIR / "legal_suffixes.json"

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_RAW_TRAIN_DIR = DATA_RAW_DIR / "train"
DATA_RAW_TEST_DIR = DATA_RAW_DIR / "test"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TRAIN_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
    "ground_truth": "train_ground_truth.tsv",
}

TEST_FILES = {
    "source1": "test_source1.tsv",
    "source2": "test_source2.tsv",
    "source3": "test_source3.tsv",
}

# Source TSV columns (verified against data/raw/*/*.tsv).
ENTITY_ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

# Ground-truth TSV columns.
GT_SOURCE1_COL = "source1_entity_id"
GT_MATCHES_COL = "matched_entity_ids"

TSV_SEP = "\t"

# Chunk size for reading multi-million-row source files. Tuned to keep a
# chunk's in-memory footprint modest (a handful of short string columns)
# well under the 16.8GB RAM budget noted in plan.md while still amortizing
# per-call overhead.
DEFAULT_CHUNKSIZE = 200_000
