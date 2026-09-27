# Business Entity Resolution

Matches every Source 1 business to its records in Source 2 / Source 3.
Final version **v2.6**: dev_val macro F0.5 0.98926 (2-fold CV, India 0.99007 /
US 0.98872), portal 0.985302.

Pipeline: cleaning → token blocking ∪ fine-tuned embedding retrieval →
57 pair features → XGBoost → three cross-encoder rounds (multilingual-e5-small)
+ a multilingual-e5-base cross-encoder on uncertain pairs → level-2 XGBoost
stacker → per-entity expected-F0.5 decisions + one-S1-per-record
post-processing → export.
Methodology and measurements: `Documentation_template.md` at the zip root.

## Layout

The code resolves every path from its own location (`src/config.py`):

```
<root>/                          # "code/" inside the submission zip
├── business_entity_resolution/
│   ├── src/                     # all code; run every command from here
│   ├── README.md
│   └── requirements.txt
├── data/raw/train/              # source1.tsv source2.tsv source3.tsv train_ground_truth.tsv
├── data/raw/test/               # source1.tsv source2.tsv source3.tsv
├── utils/validate_submission.py # the challenge's validator (student resource)
└── output/                      # written by the pipeline
```

Intermediate artifacts go to `<root>/data/processed/` (the union features alone
are ~10 GB per split).

## Environment

Python 3.12, CUDA GPU (we used an RTX 2000 Ada 16 GB for everything except the
e5-base cross-encoder, which ran on an RTX 4000 Ada 20 GB), 32 CPU threads,
64 GB RAM (full GBDT training needs 30–40 GB).

```bash
cd business_entity_resolution/src
pip install -r ../requirements.txt
```

Models (all MIT, downloaded from the Hugging Face Hub on first use, then
fine-tuned on the training data only): `intfloat/multilingual-e5-small`
(118M), `intfloat/multilingual-e5-base` (278M). No hosted APIs, no external
data.

## Reproduce end to end

Every stage caches its output and skips work already done (`--force` recomputes).
Each command shows its own progress bar; the `run_*.py` runners add a
whole-pipeline bar. Seeds are fixed (42); GPU training is not bit-exact, so a
re-run may differ in the 4th decimal.

```bash
# 1. Cleaning (~10 min)
python clean_all.py

# 2. Token blocking: IDF-selected keys per country, top-40 per S1 (~15 + 11 min)
python blocking_token.py --split train
python blocking_token.py --split test

# 3. Retriever: fine-tune multilingual-e5-small on (S1, match, hard negative) triplets
python finetune_data.py
python finetune_embeddings.py --train-dir ../../data/processed/finetune \
    --output-dir ../../data/processed/models/retriever \
    --checkpoint-dir ../../data/processed/models/retriever_ckpt

# 4. Embed + exact kNN per country, union with token candidates (k=20), 57 features,
#    dev GBDT "dev_v11" (threshold tuned on dev_val), then the final GBDT on all train
python embeddings.py --split train --k 50
python run_union_pipeline.py --force-features --run dev_v11
python train_classifier.py --mode final --gpu --features train_union_retriever_k20 \
    --dev-run dev_v11 --run final_v11
python predict.py --run final_v11 --features test_union_retriever_k20

# 5. Cross-encoder round 1 (from the retriever), re-scores pairs with GBDT >= 0.02
python cross_encoder.py export
python cross_encoder.py train
python cross_encoder.py score

# 6. Round 2 (+ France agreement pairs) and the level-2 stacker (v2.2)
python run_v22.py
# 7. Round 3 (+ synthetic French negatives/positives, French S1-S1 negatives) (v2.3)
python run_v23.py
# 8. multilingual-e5-base cross-encoder on uncertain pairs (v2.5); the budget sets how
#    many training pairs are used (benchmarked on the GPU; we used 110 min on an RTX 4000 Ada)
python run_v25.py --budget-min 110

# 9. Final stacker (run_v25.py ends with exactly this when its gate passes): writes
#    output/ (both files) and the final pair scores test_stacked_r2_r3_bag5_coh_base.parquet
python stack2.py apply --bag 5 --coherence --tag _r2,_r3 --extra-tag _base
# 10. Per-entity expected-F0.5 decisions from those scores (v2.6): rewrites
#     output/matching_results.tsv; `python expected_f.py dev` measures it on dev_val
python expected_f.py test --out-dir ../../output
cd ../.. && python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

Step 8 on a second machine instead: `python ce_base.py export` here, copy
`data/raw/train/train_ground_truth.tsv`, `data/processed/{train,test}/` and
`data/processed/cross_encoder/{train_base,eval,val_scores_r3,test_scores_r3}.parquet`
over, run `python run_e5_remote.py --budget-min 110` there, copy back
`val_scores_base.parquet` / `test_scores_base.parquet`, then run step 9.

Outputs: `output/matching_results.tsv` (one row per test S1 entity) and
`output/candidate_pairs.tsv` (the 97,191,350 union candidate pairs the models
scored), both tab-separated with `\n` line endings.

## Leakage guards

`train_classifier.split_entities` (seed 42) splits train S1 entities into
fit / early-stop / dev_val. Retriever and cross-encoder training use fit
entities only (`_assert_no_dev_val` in `cross_encoder.py`); the stacker is
trained on dev_val pairs, which neither base model saw, and evaluated by 2-fold
CV by entity. Test data is used only as unlabeled text (French agreement pairs
are pairs where two independent models agree).

## Tests

stdlib `assert`-based, no pytest needed (run from `business_entity_resolution/`):

```bash
for t in normalize address_parser blocking_token features labels evaluate export competition embeddings; do
  python tests/test_$t.py
done
```

## Design notes

- **Country is an open set.** Every stage groups by whatever string is in
  `country` (blocking partitions, feature IDF, kNN, entity split). No country
  list, no per-country table, no one-hot. France, absent from train, takes the
  identical code path at test time.
- **Metric fidelity.** `evaluate.py` implements macro F0.5 exactly as the
  problem statement defines it, averaged over every S1 entity (including
  singletons and entities with no candidates). Thresholds are tuned against
  that metric directly.
- Stage-specific decisions and the measurements behind them are in each
  module's docstring.
