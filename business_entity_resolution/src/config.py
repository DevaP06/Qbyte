"""Paths, column names, and shared constants for every implemented stage.

Currently: cleaning, token blocking, pairwise features. Extend this file as
later stages are implemented rather than pre-declaring their settings now.
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

# --- Token blocking (plan.md build-order step 2) ---------------------------
#
# Two-layer, purely statistical key selection -- no hardcoded word list
# beyond the existing legal-suffix stopwords, no country-specific branch, so
# the identical code path runs for a country never seen before (France).
# Both numbers were arrived at empirically (see blocking_token.py's module
# docstring for the measurements) after an earlier per-row "cap + fallback
# to the row's full uncapped token set" design was tried and rejected: it
# let a small number of pathological rows (addresses made entirely of
# generic geography words with no rare token at all) balloon to hundreds of
# thousands of candidates each, dominating total cost for negligible recall
# gain. This design instead drops an over-common token from consideration
# everywhere (layer 1) and never restores it -- a token blocking key
# genuinely may end up empty for a rare entity, which is an accepted, small,
# measured recall cost (~0-1.5% of entities per partition) rather than an
# unbounded one.
#
# Layer 1: a token (name or address) is dropped as a blocking key everywhere
# in a country partition once its document frequency (within that
# partition's S2-union-S3 target side) exceeds this absolute count. A token
# with zero occurrences on the target side is dropped too (see
# select_blocking_keys' docstring) -- it can never produce a match, and must
# not be treated as maximally "rare" by layer 2 below.
BLOCKING_MAX_TOKEN_DOCUMENT_FREQUENCY = 20_000

# Layer 2: among a record's surviving (non-dropped, actually-present) tokens,
# only its N rarest per side (name, address) are kept as active blocking
# keys -- bounds a single record's worst-case fan-out even when it still has
# many moderately-common survivors.
BLOCKING_KEYS_PER_SIDE = 10

# Measured recall ceiling (any shared surviving key) / recall@40 against the
# full train_ground_truth.tsv at the above two settings, from an actual
# `python blocking_token.py --split train` run:
#   US:    ceiling 99.17%   recall@40 94.69%   (52.9M candidate pairs)
#   India: ceiling 98.39%   recall@40 87.10%   (35.3M candidate pairs)
# Both clear the plan's own go/no-go bar ("target comfortably >85-90%");
# name-token-only blocking (the plan's literal text, no address tokens)
# measured at 74.9%/91.4% ceiling for India/US -- see blocking_token.py.

# Maximum candidates kept per S1 entity, ranked by summed IDF of shared
# surviving keys. plan.md's own example ("cap per entity, e.g. ~80") sizes
# the *union of token-blocking + ANN* candidates; token-blocking alone
# reusing that literal 80 would put train's total pair volume at ~176M.
# Even at 40, the actual full train run produced 88,259,486 candidate
# pairs -- itself already past the plan's stated "keep total pairs well
# under ~50-60M" sizing target, because almost every entity in both country
# partitions hits the cap exactly (mean 40.0 candidates/entity measured).
# Left at 40 rather than cut further: recall@10 was measured at
# ~91%/~81% (US/India) at similar cap settings, meaningfully below the
# recall@40 numbers above, and recall lost at blocking time can never be
# recovered downstream. See blocking_token.py's module docstring for the
# full tradeoff writeup -- this is flagged as an open decision point for
# whichever stage next measures actual runtime against this pair volume,
# not silently resolved here.
BLOCKING_MAX_CANDIDATES_PER_ENTITY = 40

# Row-batch size for the per-country scoring pass -- bounds peak memory to
# roughly (batch_size x candidates_per_row) instead of materializing one
# full country-sized score matrix at once. Measured necessary in two stages:
# a first attempt at 10,000 still OOM'd (real full-scale nnz/row runs
# ~19-26K, so summing the name+address score contributions for one 10,000-row
# batch requires scipy to allocate a ~265-million-entry buffer, ~1GB, for
# that single add -- and that's *one* batch; see BLOCKING_NUM_THREADS below
# for why concurrency multiplies this). 1,000 keeps a single batch's add
# comfortably under ~100MB.
BLOCKING_QUERY_BATCH_SIZE = 1_000

# Thread-pool width for the batch-level sparse matmuls. scipy's C-level
# sparse matmul releases the GIL, so plain threads (no multiprocessing
# complexity/pickling) measured a real ~2.5x wall-clock speedup at 8 threads
# on this machine (20 logical CPUs) with no further gain by 12 -- but that
# benchmark used one batch at a time; running the actual pipeline, each of
# these threads holds its *own* concurrently-live batch, so peak memory
# scales with num_threads x per-batch memory, not just per-batch memory
# alone. Kept at 4 (rather than the benchmarked 8) to leave more headroom
# after the OOM above, trading some wall-clock time for a larger safety
# margin.
BLOCKING_NUM_THREADS = 4

CANDIDATES_DIR = DATA_PROCESSED_DIR / "candidates"

# --- Pairwise features (plan.md build-order step 3) ------------------------

FEATURES_DIR = DATA_PROCESSED_DIR / "features"

# Candidate pairs per feature chunk. Chunks are cut on S1-entity boundaries
# (never mid-entity, so per-S1 context features see the entity's whole
# candidate list), so actual chunk sizes run slightly over this. At ~40
# candidates/entity, 1M pairs keeps every gathered sparse slice and string
# list in a chunk to a few hundred MB.
FEATURE_CHUNK_PAIRS = 1_000_000

# Hashed feature space for name character trigrams. Stateless hashing (no
# fitted vocabulary) so train and test -- including France's unseen
# vocabulary -- go through the identical code path; 2^21 buckets keeps
# collisions rare for the few hundred thousand distinct trigrams across
# Latin + Devanagari/Kannada names.
FEATURE_CHAR_NGRAM_HASH_SIZE = 2**21

# --- Classifier (plan.md build-order step 4) --------------------------------

MODELS_DIR = DATA_PROCESSED_DIR / "models"

# Entity-level split (never within an S1 entity), stratified by country:
# dev_val is held out for threshold tuning + reported F_0.5; the early-stopping
# set is carved from what remains, so dev_val never influences training.
DEV_VAL_FRACTION = 0.15
EARLY_STOPPING_FRACTION = 0.05
SPLIT_SEED = 42

# Stock binary objective, no positive re-weighting: the train positive rate
# among candidates (~8%) is not extreme for a GBDT, and F_0.5 favours
# precision -- up-weighting positives pushes the other way. The macro-F_0.5
# threshold search absorbs the class prior instead. bagging_fraction 0.5
# halves per-iteration cost on ~75M training rows.
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 127,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "seed": SPLIT_SEED,
    "verbose": -1,
}

# XGBoost equivalent, for GPU training (train_classifier.py --gpu). Leaf-wise
# growth (lossguide + max_leaves) mirrors LightGBM's tree shape; the rest
# mirrors the LightGBM settings above one-for-one where a counterpart exists.
# min_child_weight is a hessian sum, not a row count: at ~8% positives a leaf
# of 500 rows has a hessian of roughly 500 * p(1-p) ~ 20-40, so 20 is the
# closest analogue of min_data_in_leaf=500.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.1,
    "grow_policy": "lossguide",
    "max_depth": 0,
    "max_leaves": 127,
    "min_child_weight": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.5,
    "reg_lambda": 1.0,
    "max_bin": 256,
    "seed": SPLIT_SEED,
}

# The first full-data GPU dev run (70.6M fit pairs) was still improving at
# 2000 rounds -- it hit the cap without early stopping -- so the cap sits
# well above that; early stopping is what actually ends training.
MAX_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 50

# --- Prediction + submission (plan.md steps 5 + 9) ---------------------------

PREDICTIONS_DIR = DATA_PROCESSED_DIR / "predictions"
OUTPUT_DIR = REPO_ROOT / "output"
