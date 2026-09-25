# Status & where to pick up next

_Last updated: 2026-09-26. Read this first when resuming on a new machine._
Full plan and rationale: [plan.md](plan.md). Rules: [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).
How to run each stage: [business_entity_resolution/README.md](business_entity_resolution/README.md).

## Where we are

| plan.md step | State |
|---|---|
| 1. Cleaning | ✅ done |
| 2. Token blocking | ✅ done (train + test candidates built) |
| 3. Features + labels | ✅ done (43 features, train + test) |
| 4. Classifier + threshold | ✅ done — XGBoost on GPU, dev run finished |
| 5. Predict + post-process + export | ✅ code done, validator-passing on the full test set |
| **Final model + first leaderboard submission** | ⏭ **next** |
| 6–7. Embedding retrieval (+ SageMaker fine-tuning) | 🟡 fine-tuning data + scripts ready, not run |
| Phase A/B improvements (below) | not started |

**Best dev_val macro F0.5: 0.9469** (`data/processed/models/dev_gpu`, XGBoost CUDA,
early-stopped at 3,871 rounds, threshold 0.71). Target: top leaderboard score is **0.988**.

| | macro F0.5 | candidate ceiling* |
|---|---|---|
| India | 0.9210 | 0.9460 |
| US | 0.9642 | 0.9809 |
| **ALL** | **0.9469** | **0.9669** |

\* F0.5 of a *perfect* classifier on our candidates, i.e. the most any classifier could reach.

## Where the remaining score is lost (measured on dev_val)

| Cause | Loss | Notes |
|---|---|---|
| **Blocking** — a true match never became a candidate | **0.037** (70%) | 68K dev_val entities (21%) miss ≥1 true match |
| Classifier false positives | 0.008 | incl. 871 true singletons given a match |
| Classifier false negatives | 0.008 | true match was a candidate but scored below the threshold |

Share of true pairs that reach the candidate list: **US 94.7%, India Latin-script
90.2%, India Devanagari/Kannada 73.1%.** For 98–99% of true pairs, S1 and the
candidate *do* share a token, but the match is ranked below the top-40 cut.

**Conclusion:** past 0.967 the only route is better candidates (Phase A). More
boosting rounds took 2000 → 3871 rounds for just +0.0004, so the classifier is
saturated on the current features (Phase B adds new ones).

## 1. Immediate next step: final model → first submission

On the new PC, after cloning and copying the `data/` folder (see Setup):

```powershell
cd business_entity_resolution\src
python train_classifier.py --mode final --gpu --dev-run dev_gpu --run final_gpu   # 3871 rounds, all 88M pairs
python predict.py --run final_gpu                                                 # ~10 min -> output\*.tsv
cd ..\..
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test --check-ids
```

On `PASS`, upload `output/matching_results.tsv` to the portal. Record the score here.
**Compare it with the dev_val 0.947**: the difference is mostly France (15% of test, never in train).

Already verified end to end with the earlier dev model: 1,732,544 rows; the
validator passed with `--check-ids`. 5.36M pairs were above the threshold, and
5.32M were left after conflict resolution. Predicted singletons: France 6.7%,
India 8.2%, US 5.9% (true train rate: 5.6%).

## 2. Phase A: candidate recall (the 0.037 bucket)

Order matters: A1 + A2 change cleaning, so batch them. Each full rebuild
(clean → block → features → train) takes ~1–1.5 h.

**A1. Abbreviation normalization.** Add a hand-written map, e.g.
`src/resources/address_abbreviations.json`, applied in `normalize.py` /
`address_parser.py`:
- US: `rd→road`, `st→street`, `ave→avenue`, `blvd→boulevard`, `dr→drive`, `ln→lane`
- French: `r./r→rue`, `av./av→avenue`, `bd→boulevard`, `st.→saint`, `ste→sainte`, `pl→place`
- India: `ngr→nagar`, `opp→opposite`, `nr→near`

It must be hand-written, not downloaded (see Rules). **Watch out for `st`:** it
means street in `main st` but saint in `st.-nazaire`. Decide by position.

**A2. Transliteration.** Add Latin tokens for Devanagari/Kannada/Tamil names
with `anyascii` (ISC license, local library). Use them as extra blocking keys
and for name features. This targets the 73% bucket.

**A3. Fine-tune the retrieval model on SageMaker.** Everything is ready:
- `data/processed/finetune/`: 1.5M train rows (510K are true matches blocking
  missed) + 20K eval rows. Built from `fit` entities only; dev_val is never
  included, so scores stay honest.
- `src/finetune_embeddings.py`: standalone; runs on SageMaker or any GPU. The
  model is `intfloat/multilingual-e5-small` (MIT, 118M params). A CPU smoke test
  on 2K rows took eval triplet accuracy from **0.918 to 0.974**.
- `sagemaker/launch_finetune.py`: uploads the data, runs the job and downloads
  the model. No AWS CLI needed. Run it from its own virtualenv:

```powershell
python -m venv .venv-sagemaker
.venv-sagemaker\Scripts\pip install -r business_entity_resolution/sagemaker/requirements.txt
$env:SAGEMAKER_SUPPRESS_V2_WARNING = "1"
.venv-sagemaker\Scripts\python business_entity_resolution/sagemaker/launch_finetune.py --role <SageMaker role ARN> --region <region> --wait
```

It needs AWS credentials (env vars or `~/.aws/credentials`) and a SageMaker
execution role with S3 access. The model lands in
`data/processed/models/retriever/`. Or, with a good local GPU, skip SageMaker:
`python src/finetune_embeddings.py --train-dir ../../data/processed/finetune --output-dir ../../data/processed/models/retriever`.

**A4. Embedding retrieval (not written yet).** New module `src/embeddings.py`:
- Embed every record with `finetune_data.record_texts(split)`, the **same**
  `"query: "` prefix, fp16 and L2-normalized. Deduplicate identical texts first.
- Take the exact top-k per country with torch matmul on the GPU (no faiss
  needed; the largest shard of ~6M × 384 fp16 fits in 16 GB).
- Union the results with the token candidates.
- Run the off-the-shelf e5 model first, then the fine-tuned one, and compare recall.

**A5. Re-rank a larger pool.** Take ~200 token candidates + ~50 embedding
candidates, score them with a cheap ranker, and keep the top 40–50.
**Target: >99% of true pairs in candidates.**

Measure after each step: blocking recall (the `blocking_token.py --split train`
log, or `labels.py --features train`), then dev_val F0.5. Keep only what helps.

## 3. Phase B: classifier features (the 0.016 bucket)

- **Reverse-rank:** the ground truth says each S2/S3 id belongs to at most one
  S1. Add a feature for "is this S1 the candidate's best match among all S1s
  that have it as a candidate?", and its rank. This works best as a two-stage
  model: stage-1 scores → reverse-rank features → stage-2.
- **Coherence:** an S1's true matches are noisy copies of each other. Add the
  similarity of a candidate to the S1's other top candidates, especially those
  from the other source (S2 vs S3).
- **Embedding cosine** (from A4) and **transliterated-name similarities** (from A2).
- Post-processing already keeps only one S1 per S2/S3 id (+0.0004 on dev_val).

## 4. Phase C/D: France check and submission package

- Run `train_classifier.py --run <name> --gpu --loco` on full data. So far it
  has only run on a 100K sample: holding out India cost 0.134 (driven by
  non-Latin names: −0.32 on those entities vs −0.07 on Latin-only ones);
  holding out US cost 0.038. France, which is Latin-script, likely costs 4–7
  points. Re-check after Phase A.
- Package: pin `requirements.txt` (add torch/sentence-transformers once A4
  lands), fill in `Documentation_template.md` with the measured numbers
  (recall, ceiling, F0.5, LOCO gap, models + licenses), and zip in the
  structure from PROBLEM_STATEMENT.md.

## Setup on a new machine

```powershell
git clone <repo-url> Qbyte; cd Qbyte
robocopy <old-pc>\Qbyte\data .\data /E /MT:16 /R:2 /W:5     # 19.4 GB, all gitignored
pip install -r business_entity_resolution/requirements.txt
python -c "import xgboost as xgb; print(xgb.build_info()['USE_CUDA'])"   # must print True
# only for fine-tuning / embeddings (not in requirements.txt yet):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install sentence-transformers==3.4.1 transformers==4.48.3 datasets==3.2.0 accelerate==1.3.0
```

Tests: `for t in normalize address_parser blocking_token features labels evaluate export; do python business_entity_resolution/tests/test_$t.py; done` (all pass).

What the copied `data/` holds:

| Folder | Contents |
|---|---|
| `raw/` | Original TSVs + `train_ground_truth.tsv` |
| `processed/{train,test}/` | Cleaned sources |
| `processed/candidates/` | Token blocking: 88.3M train / 69.3M test pairs |
| `processed/features/{train,test}/` | 43 features per pair |
| `processed/models/dev_gpu/` | Best dev model + `val_predictions.parquet` for error analysis |
| `processed/finetune/` | SageMaker fine-tuning data |

## Rules checklist (PROBLEM_STATEMENT.md)

- ✅ Allowed:
  - XGBoost (Apache-2.0) and LightGBM (MIT)
  - MIT/Apache embedding models ≤8B params (e5-small: 118M)
  - Fine-tuning on the provided train data only
  - SageMaker/S3 as compute and storage
  - Hand-written normalization maps
  - Local libraries like `anyascii`
  - Unlabeled test text (IDF, embeddings)
- ❌ Not allowed:
  - Hosted APIs: embedding/LLM APIs, geocoding, business registries, scraping
  - Internet datasets (e.g. downloaded abbreviation lists)
  - Models >8B params or with non-MIT/Apache licenses
- `country` is an open set everywhere: no `{US, India}` lists, no one-hot, no
  per-country tables. France must use the same code path.

## Gotchas learned

- **The PyPI LightGBM wheel has no GPU support**, so `--gpu` switches to XGBoost CUDA.
- **Output files must use `\n` line endings.** Windows `\r\n` would glue `\r`
  onto the last id of each row. `export.py` enforces this, and a test checks it.
- **Memory:** full training needs ~30–40 GB host RAM (`--fit-entity-frac 0.3`
  helps on smaller machines). Train blocking peaks ~25 GB.
- **Progress bars** redraw in place in a terminal. Piping through `Tee-Object`
  turns them into one line every 30 s.
- **Sample runs** (`--features train_sample100000`) score only the sampled
  entities, so they aren't comparable to full-data numbers.
- **`predict.py` score cache** refreshes automatically when the model or
  features are newer, so copying with robocopy (which keeps timestamps) is safe.
