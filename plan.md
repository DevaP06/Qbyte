# Business Entity Resolution — Full Pipeline Plan

## Context

"ML Challenge 2026: Business Entity Resolution", in
`6ab10eb3b23ba_student_resource/student_resource/`. Source 1 is a
deduplicated reference set of business entities; for every Source 1 entity we
must find all matching records in Source 2 and Source 3 (0, 1, or many
matches). Scored by macro-averaged F_0.5 per Source 1 entity — precision
weighted 2x over recall, so false merges are punished hard and correctly
predicting "no match" (singleton) matters.

The user supplied a brainstorm doc with a lot of research-grade techniques
per task (GNNs, Ditto-style contrastive learning, optimal-transport
cross-attention alignment, Rotom augmentation, hierarchical tree-edit
distance, full graph clustering). **Directive: don't adopt these just
because they're in the brainstorm — take the useful ideas, but every
decision below is chosen and justified by what's actually feasible and
evidence-backed from the data**, not by "it's cutting-edge." Where a
brainstorm idea is rejected or scoped down, the reason is stated explicitly.

**Grounding facts (verified directly against the files, not assumed):**

| Fact | Value |
|---|---|
| train rows | S1 2,206,821 · S2 5,034,616 · S3 5,285,603 · GT 2,206,821 |
| test rows | S1 1,732,544 · S2 4,887,273 · S3 5,082,316 (no ground truth) |
| train country mix | US 60.0% · India 40.0% |
| test country mix | India 46.8% · US 38.3% · **France 15.0%** (unseen at train time) |
| GT singleton rate | 5.58% of S1 entities have no match |
| GT matches/entity | mean 3.46, median 3, max 11 |
| **Cross-country GT pairs** | **0 out of 7,638,365** matched pairs ever cross a country boundary |
| **Same S2/S3 id claimed by >1 S1** | **0 out of 7,638,365** GT pairs |
| empty `business_address` | 0% in S1, ~3% in S2/S3. `business_name` never empty |
| postal codes | **absent from the corpus** — not usable as a feature/component |
| address structure | US addresses regular (~2 commas: street/city/state); India addresses irregular (3–8+ commas, landmarks, reordering) |
| non-ASCII names | ~15% of a 500K India+US Source-2 sample contain Devanagari/Kannada script |
| dominant name tokens | `limited, llc, private, inc, ltd, com, center, pvt, services, corp, group, india, holdings, care, associates, llp` — mostly legal suffixes + generic words, not distinctive |
| current machine | RTX 3050 Laptop (6GB VRAM), 16.8GB RAM, torch cu126 present; lightgbm/xgboost/faiss/sentence-transformers/rapidfuzz **not yet installed** |

**Hard constraints (README + validator):**
- No external DB/API/geocoding/internet lookups — fully self-contained, no
  hardcoded country list (France must work via the same code path as
  US/India, never an `if country == "France"` branch).
- Final model(s) ≤8B params, MIT/Apache-2.0 licensed (applies to the neural
  components — bi-encoder, optional cross-encoder; LightGBM is unaffected
  but note this distinction in the write-up).
- Two outputs: `output/candidate_pairs.tsv` (must be the *exact* final
  candidate set the classifier ran over, not an earlier raw blocking pass)
  and `output/matching_results.tsv` (the only scored file). Matches must be
  a subset of candidates; no dup IDs, no self-matches, every test S1 entity
  present exactly once. Gate: `utils/validate_submission.py`.
- Submission needs runnable code + `requirements.txt` + filled
  `Documentation_template.md`. Nothing exists yet beyond the validator.

**Timing:** no dedicated GPU/server until tomorrow (today's machine has only
a modest 6GB laptop GPU). Plan is sequenced so **today is 100% CPU-feasible**
(normalization, token blocking, feature engineering, classical similarity
metrics, LightGBM at reduced/sampled scale) and **GPU-heavy work is queued
for tomorrow** (full-corpus bi-encoder embeddings, FAISS index build,
full-scale training, optional cross-encoder reranker). Re-check the sizing
table below against tomorrow's actual server specs before committing to a
runtime budget.

## Technique decisions (evidence-based, not brainstorm-by-default)

Rejected outright as infeasible/unjustified overhead at this scale and
compute budget: GNN/RGCN/GAT structural features, full Ditto-style
contrastive representation learning, optimal-transport/cross-attention
alignment, Rotom-style self-supervised augmentation, running a
transformer cross-encoder over the full candidate set. None of these are
needed to hit a strong F_0.5, and each adds real engineering/compute cost
with uncertain payoff at 10M+ rows.

### Task 1+3 — Candidate generation & retrieval

1. **Normalization**: lowercase, Unicode NFKC, strip URL/domain-style noise
   (seen literally: `wilfordhancock.com`), expand+strip legal-suffix
   abbreviations (Pvt/Private, Ltd/Limited, Corp/Corporation, Inc, LLC, LLP)
   via a small hand-authored `legal_suffixes.json` map (static domain
   knowledge, not an external lookup), strip leading punctuation noise,
   ASCII-fold Latin text; keep the original string (not folded) for
   embedding so non-Latin scripts stay intact for the bi-encoder.
2. **Country is a hard partition, not a soft feature.** Checked all
   7,638,365 GT pairs directly: zero ever cross a country boundary. A hard
   filter costs no measurable recall and shrinks blocking/embedding/index
   work substantially. It generalizes to France for free — partitioning is
   "group by whatever string is in `country`," which requires no prior
   knowledge of which values exist; France simply forms its own group at
   test time through the identical code path. One defensive check: log the
   distinct country-key groups and sizes per split at pipeline startup, so
   a silent variant spelling (e.g. "USA" vs "US") producing an orphan
   micro-group is caught rather than silently losing recall.
3. **Primary blocking = inverted-index token blocking** on normalized name
   tokens (with the dominant generic/legal tokens like `services`,
   `private`, `india`, `ltd`, `inc` excluded as blocking keys — they carry
   no discriminative signal for retrieval, though they still feed features
   later). Cheap, CPU-only, does most of the recall/reduction work.
4. **Secondary blocking = dense retrieval** (bi-encoder embeddings + FAISS
   ANN), additive on top of token blocking, to recover cases token
   blocking misses: typos, transliteration, non-Latin scripts, word
   reordering. GPU-dependent — tomorrow. Use a small **multilingual**
   MIT/Apache-licensed sentence encoder (not an English-only one), because
   ~15% of names are non-Latin script and France text needs to embed
   meaningfully too — verify the exact license on the model card before
   locking in a specific checkpoint at implementation time.
5. **MinHash/LSH is skipped as a separate blocking stage**, not adopted
   just because it's in the brainstorm. It targets the same
   near-duplicate/typo failure mode that dense ANN retrieval already
   covers (and ANN covers it better, since it also handles semantic
   reordering and cross-script variance that pure n-gram MinHash can't).
   Its underlying signal isn't thrown away — it's relocated to a cheap
   *feature* (character trigram Dice, computed only on the already-small
   candidate set) instead of a full-corpus blocking pass. This is a
   go/no-go gate, not a permanent ban: if the measured recall ceiling after
   {token blocking ∪ ANN} shows a specific gap attributable to
   character-level noise, add MinHash then, targeted at the measured gap.
6. Union token-blocking candidates + ANN candidates per S1 entity, cap per
   entity (e.g. ~80) if a shard is pathologically large. The exact
   surviving set (after Task 5 post-filtering) is written once and reused
   identically for `candidate_pairs.tsv` and as the classifier's input.

### Task 2 — Feature engineering

Classical, cheap, and precision-focused — no exotic representation
learning:
- Name & address (separately): token Jaccard, normalized
  Levenshtein/edit distance, Monge-Elkan, character trigram Dice — via
  `rapidfuzz` (MIT), whose batch C routines are the difference between
  tractable and not at tens-of-millions-of-pairs scale.
- TF-IDF cosine similarity (char n-gram + word n-gram) for name and
  address.
- Bi-encoder cosine similarity **reused** from the Task 1/3 ANN step, never
  recomputed.
- IDF-weighted token overlap (down-weights generic tokens like Pvt/Ltd/
  Services/India, up-weights rare/brand tokens) from the same TF-IDF
  vectorizer — cheap, directly targets precision.
- Heuristic structured address comparison instead of full tree-edit
  distance: addresses have no fixed schema (comma count ranges 2–8+, most
  variable in India), so there's no real tree to align in the first place —
  any "tree" would itself be a heuristic guess. Instead: positional
  comma-split into head (street-like) / tail (region-like) / middle
  (bag-of-tokens), compare corresponding pieces. Captures the same
  "compare like-with-like" idea at much lower implementation risk, and
  degrades gracefully on the ~3% empty-address rows.
- Metadata: address-present flag, name/address length (diff + ratio), and
  a standalone-digit-token overlap feature that explicitly catches the
  "Suite 100 vs Suite 101" false-merge case (conflict = both sides have
  digit tokens and the sets are disjoint). `country_exact_match` is *not*
  a feature — since blocking is already hard-partitioned by country, it's
  a constant `1` inside the candidate set and carries no signal.
- Postal code / landmark features are **not implemented** — confirmed
  absent from the data, so building them would be dead code.

### Task 4 — Classification

- **Primary model: LightGBM** (or XGBoost) on the Task 2 feature vectors.
  Practical at tens-of-millions of candidate pairs; a transformer
  cross-encoder over that volume isn't viable even on GPU.
- Baseline first: stock binary objective + class-imbalance handling
  (candidate sets skew heavily negative — mean 3.46 true matches per
  entity vs. up to ~80 candidates), get a working end-to-end model before
  adding complexity.
- **Custom asymmetric objective** (weighted binary cross-entropy / focal
  loss variant) penalizing false positives more than false negatives,
  matching F_0.5's 2x precision weighting — layered in as an improvement
  pass once the baseline works, with the weight tuned directly against
  validation macro-F_0.5, not logloss.
- **Threshold tuning**: single global scalar threshold on held-out train
  data, maximizing macro F_0.5 with the README's exact per-entity
  semantics (empty-predicted/empty-true → 1.0; any prediction on a true
  singleton → 0.0). A *global* threshold, not a per-country lookup table —
  a lookup keyed by literal country string has no entry for "France" and
  would crash or silently misbehave. If per-country calibration proves
  necessary, prefer a self-computing procedure (e.g. per-country
  percentile-rank of that country's own test-time scores) over a fixed
  pre-fit table, since it generalizes to France without needing a
  France-specific value in advance.
- **Optional stretch, gated, not default**: small MIT/Apache cross-encoder
  reranking only the classifier's top-K (e.g. top-3) shortlist per S1
  entity. Go/no-go: only add it if validation shows the GBDT is measurably
  leaving precision on the table specifically on close-to-threshold pairs,
  and only after the baseline pipeline already works end-to-end.

### Task 5 — Post-processing guardrails

- Filter to the tuned threshold; nothing-above-threshold → empty
  `matched_entity_ids`.
- **Conflict resolution, scoped down from the brainstorm's full graph
  clustering**: since the GT evidence shows 0 of 7,638,365 pairs ever have
  the same S2/S3 id claimed by two different S1 entities, resolve
  conflicts by keeping only the highest-scoring edge per S2/S3 id (S1 side
  stays many-to-one, since the true mean is 3.46 matches per entity — a
  full 1:1 bipartite match or connected-components clustering across all
  three sources would wrongly force one-to-one on the S1 side too, or
  conflate S1's role as a fixed reference with a mutual-clustering problem
  the task doesn't actually ask for). This is a cheap sort + de-dupe by
  id, and expected to be a near-pure precision win given the zero-exception
  evidence.
- Export `candidate_pairs.tsv` and `matching_results.tsv` from the same
  final candidate set via one shared writer function (guarantees identical
  formatting/edge-case handling, and that the shipped candidate file is
  exactly what the classifier saw). Log recall ceiling, reduction ratio,
  and per-country F_0.5 breakdown as standing diagnostics.

## Repository structure

```
code/business_entity_resolution/
├── requirements.txt
├── README.md
└── src/
    ├── config.py                # paths, hyperparams, cache dirs, column names
    ├── io_utils.py               # tsv read/write, chunked iterators, cache-or-build helper
    ├── resources/legal_suffixes.json
    ├── normalize.py              # cleaning, tokenization, digit-token extraction
    ├── address_parser.py         # heuristic comma-split component guesser
    ├── blocking_token.py         # inverted index + token candidate generation
    ├── embeddings.py             # bi-encoder batch inference, fp16 memmap cache (GPU, day 2)
    ├── ann_index.py              # FAISS build/query per country shard (GPU, day 2)
    ├── candidate_generation.py   # union+cap, canonical candidate I/O, writes candidate_pairs.tsv
    ├── tfidf_models.py           # fit/persist char+word TF-IDF vectorizers
    ├── features.py               # candidate pairs -> feature matrix, chunked
    ├── labels.py                 # join candidates with train ground truth
    ├── custom_objective.py       # asymmetric grad/hess for LightGBM
    ├── train_classifier.py       # training, CV, persistence, feature importance
    ├── threshold_tuning.py       # macro-F0.5-optimal threshold search
    ├── reranker.py                # optional cross-encoder stretch, gated
    ├── postprocess.py            # threshold filter + conflict resolution
    ├── export.py                  # shared writer for both output formats
    ├── evaluate.py                # local F0.5, recall ceiling, reduction ratio, LOCO diagnostics
    └── pipeline.py                 # CLI entrypoint, one subcommand per stage, --force to recompute
```

Every stage caches its output to disk (Parquet/npy) and is independently
re-runnable — expensive steps (embeddings especially) are never silently
recomputed.

## Train/validation split & the France risk (concrete plan, not just a caveat)

1. Split at the `source1_entity_id` level (never split within an entity)
   into `dev_train` (~85%) / `dev_val` (~15%), stratified by country to
   mirror train's own US/India mix.
2. Build a second, reweighted view of `dev_val` resampled to test's
   India:US ratio, to at least correct for the train/test mix shift on the
   two countries we can measure.
3. **Leave-one-country-out (LOCO) stress test** — the concrete answer to
   "what do we do about France," not just a documented limitation: train a
   diagnostic model on India-only pairs, evaluate on US-only `dev_val` (and
   the mirror direction). This measures the expected F_0.5 drop when the
   classifier has seen zero labeled examples from a country — a quantified
   proxy for France's exact situation at test time. Report this LOCO gap
   in the write-up. Acceptance bar: since features are country-agnostic by
   design (no one-hot, no gazetteer, pure string-similarity + embedding
   cosine), the LOCO gap should be small — if it isn't, that's a signal the
   features implicitly overfit to country-specific address idioms and need
   revisiting before trusting the France segment.
4. Final model: retrain on all of train (dev_train + dev_val) after
   threshold/hyperparameters are locked, carrying the locked global
   threshold forward unchanged.

## Sizing / performance risks

| Step | Estimate | Risk | Mitigation |
|---|---|---|---|
| Texts to embed | ~24.2M combined (train+test) | — | dedup identical name+address strings before encoding |
| Embedding runtime | biggest single wall-clock cost in the pipeline (hours, GPU-dependent) | Yes | short max sequence length, fp16, dedup, run as an overnight background job decoupled from feature/model iteration |
| Embedding storage | ~10–20GB on disk (fp16, memmapped) | Disk, not RAM | shard by (split, country, source); never load a full split at once |
| FAISS index memory | largest country shard could be several GB as flat float32 | RAM pressure | scalar-quantized (SQ8) IVF index for large shards, plain HNSW for smaller ones (e.g. France) |
| Candidate pairs after blocking | not knowable without running it; capped by design (~40 token + ~60 ANN per entity worst case) | Feature-stage memory if caps too loose | run a 5–10% dry run first to measure real reduction ratio before committing to full-corpus feature computation; target keeping total pairs well under ~50–60M |
| Feature computation | tractable via rapidfuzz batch routines if chunked | Memory if not chunked | process in 500K–1M row chunks, write incrementally to partitioned Parquet |
| LightGBM training | standard scale for tens of millions of rows × ~20 features | Low | CPU is fine; keeps GPU free for embeddings |

## Build order

**Today (CPU-only):**
1. `normalize.py`, `legal_suffixes.json`, `address_parser.py` — unit-test
   against real messy examples pulled from the data before scaling up.
2. `blocking_token.py` — run on full train data; measure reduction ratio,
   avg candidates/entity, and **recall ceiling against
   `train_ground_truth.tsv`**. First go/no-go checkpoint (target
   comfortably >85–90% before moving on).
3. `features.py` (classical features only; embedding-cosine column left
   as a placeholder until tomorrow) + `labels.py`.
4. `train_classifier.py` baseline (stock objective) + `threshold_tuning.py`
   on the token-blocking-only candidate set — first real scored milestone,
   plus the LOCO diagnostic run.
5. `postprocess.py` + `export.py`; run `utils/validate_submission.py`
   end-to-end on this CPU-only baseline to confirm the whole pipeline is
   wired correctly before GPU work starts.

**Tomorrow (GPU/server available):**
6. `embeddings.py` + `ann_index.py` — encode all corpora, build
   country-partitioned FAISS indices, union ANN candidates into step 2's
   set; re-measure the recall-ceiling gain from adding ANN (this number is
   also the concrete evidence for the MinHash go/no-go decision).
7. Re-run `features.py` with the real bi-encoder cosine filled in, retrain
   with the custom asymmetric objective, re-tune the threshold.
8. Evaluate the optional cross-encoder `reranker.py` against its go/no-go
   criteria.
9. Full run on `dataset/test` → `output/candidate_pairs.tsv` +
   `output/matching_results.tsv`.
10. Validate (`utils/validate_submission.py`, including a `--check-ids`
    pass despite its memory cost), pin `requirements.txt`, write
    `code/business_entity_resolution/README.md` and fill in
    `Documentation_template.md` with the quantitative diagnostics gathered
    along the way (recall ceiling, reduction ratio, F_0.5, LOCO gap,
    confirmed model licenses/param counts).

## Verification

- After step 2 and again after step 6: recall ceiling + reduction ratio on
  train.
- After step 4/7: macro F_0.5 (+ LOCO gap, per-country breakdown) on
  held-out train validation.
- Final gate: `python3 utils/validate_submission.py --matching
  output/matching_results.tsv --candidate output/candidate_pairs.tsv
  --test-dir dataset/test` must pass clean.
