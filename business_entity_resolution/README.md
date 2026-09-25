# Business Entity Resolution

Source package for the entity resolution solution (blocking + matching).
See `../plan.md` for the full pipeline plan.

## Status

**Implemented** (plan.md build order steps 1–5):

| Step | Module | Output |
|---|---|---|
| 1. Cleaning | `clean_all.py` (`normalize.py`, `address_parser.py`, `io_utils.py`) | `data/processed/<split>/source{1,2,3}.parquet` |
| 2. Token blocking | `blocking_token.py` | `data/processed/candidates/<split>/token_blocking.parquet` |
| 3. Features + labels | `features.py`, `labels.py` | `data/processed/features/<split>/part-*.parquet` |
| 4. Classifier + threshold | `train_classifier.py`, `threshold_tuning.py`, `evaluate.py` | `data/processed/models/<run>/` |
| 5. Predict + post-process + export | `predict.py`, `postprocess.py`, `export.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

Best dev_val macro F0.5 so far: 0.9465 (XGBoost on GPU, 2000-round cap;
candidate-set ceiling 0.9669).

**In progress:** retrieval-embedding fine-tuning (`finetune_data.py`,
`finetune_embeddings.py`, `../sagemaker/launch_finetune.py` — data export
and training script done, retrieval integration not yet).
**Not yet implemented:** embedding retrieval in blocking (steps 6–7),
optional reranker.

## Usage

Run from `business_entity_resolution/src/`. Every stage caches its output and
skips if present; pass `--force` to recompute.

```bash
pip install -r ../requirements.txt

python clean_all.py                       # ~10 min, all six source files
python blocking_token.py --split train    # ~15 min; prints recall ceiling vs ground truth
python blocking_token.py --split test     # ~11 min
python features.py --split train          # ~13 min, 88M pairs
python features.py --split test           # ~12 min, 69M pairs
python labels.py --features train         # label coverage report (optional)
python train_classifier.py --run dev_gpu --gpu [--loco]   # dev split, threshold, per-country (+ LOCO) F0.5
python train_classifier.py --mode final --gpu --dev-run dev_gpu --run final_gpu   # all train, locked threshold
python predict.py --run final_gpu          # -> ../../output/*.tsv
cd ../.. && python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

`--gpu` trains XGBoost on CUDA (the PyPI LightGBM wheel has no GPU support);
without it, LightGBM on CPU. Full-data training needs ~30–40GB host RAM; add
`--fit-entity-frac 0.3` on smaller machines.

Quick iteration on a sample: `features.py --split train --sample-entities 100000`
then `train_classifier.py --features train_sample100000 --run smoke`.
Timings measured on a 32-thread / 64GB machine.

Tests (stdlib `assert`-based, no pytest dependency):

```bash
for t in normalize address_parser blocking_token features labels evaluate export; do
  python tests/test_$t.py
done
python tests/smoke_test_real_data.py   # real-data sample checks for cleaning
```

## Design notes

- **Country is an open set.** Every stage groups by whatever string is in
  `country` (blocking partitions, feature IDF, entity split) — no country
  list, no per-country table, no one-hot. France, absent from train, takes
  the identical code path at test time.
- **Metric fidelity.** `evaluate.py` implements macro F0.5 exactly as
  PROBLEM_STATEMENT.md defines it, averaged over *every* S1 entity —
  including singletons and entities blocking gave no candidates — and the
  threshold is tuned against that metric directly, never a proxy.
- Stage-specific decisions, deviations from plan.md, and the measurements
  behind them are documented in each module's docstring.
