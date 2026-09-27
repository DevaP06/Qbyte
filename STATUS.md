# Status & where to pick up next

_Last updated: 2026-09-27 07:50. Read this first when resuming._
Plan: [plan.md](plan.md) · Rules: [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) ·
How to run each stage: [business_entity_resolution/README.md](business_entity_resolution/README.md)

## Scoreboard

| Version | Tag | What changed | dev_val F0.5 | Portal |
|---|---|---|---|---|
| v0.1.0 | – (commit `150a178`) | token blocking + 43 features + XGBoost | 0.9469 | 0.9378 |
| **v1.0.0** | `v1.0.0` | + fine-tuned e5 embedding retrieval (k=20) + 5 embedding features | **0.9799** | **0.9724** |
| **v1.1** | `v1.1.0` | cycle 1: competition + street-core + acronym features (57 features) | **0.9857** | **0.978** |
| **v2.0** | `v2.0` | + cross-encoder (from the retriever) on pairs with GBDT ≥ 0.02, linear blend | **0.9872** | **0.981238** |
| v2.1 | – (built as the gate fallback) | + level-2 stacker (XGBoost over GBDT + CE scores + 14 pair features), 2-fold CV by entity | 0.9887 (CV) | – |
| **v2.2** | `v2.2` | CE round 2 (new fit entities + France agreement pairs) → stacker; `python src/run_v22.py` | **0.9888 (CV)** | _pending_ |

Measured and rejected (no significant gain): bipartite assignment (+0.00000), label-free
France threshold (−0.0015 on US-as-unseen), pseudo-labeled GBDT (+0.0008), reverse retrieval
(≤ +0.0002), larger retrieval K (≤ +0.0009 total), stacker with all 57 features (+0.0001).

Leaderboard (≈24 h left): top 3 = 0.990, top 50 ≥ 0.988. **Goal: top 10 (≈ 0.990+).**
That needs France ≈ 0.985: v1.1 implies France ≈ 0.927 vs India 0.9865 / US 0.9852,
so France alone costs ≈ 0.009 of portal score.

v1.0.0 dev_val by country: India 0.9804 / US 0.9795, candidate ceiling 0.9992.
The portal is lower than dev because of **France** (15% of test, no train labels):
implied France ≈ 0.92 vs ≈ 0.98 for India/US.

## Where the remaining loss is (v1.0.0, exact)

Measured with `error_analysis.py` + the debug API: each cause's wrong decisions
were flipped to correct and dev_val was rescored (upper bound per fix).

| Cause | Gain if fixed |
|---|---|
| **FN: name-only candidate (no address)** | **+0.0087** |
| FN: typos / suffix / partial address | +0.0030 |
| **FP: name-only candidate** | **+0.0019** |
| FN: different / DBA trade name | +0.0017 |
| FP: near-duplicate distractor | +0.0017 |
| FP: same address, different business | +0.0012 |
| FP: house-number conflict | +0.0009 |
| FN: non-Latin name | +0.0004 |
| Retrieval misses (MISS) | ~+0.0009 |
| **All classifier errors** | **+0.0190** (ceiling 0.9992) |

**Root cause:** the model scores each pair in isolation. A name-only record is a
candidate of ~37 S1 entities, and whether it belongs to *this* one depends on
the rivals. On dev_val name-only pairs, "this S1 is the best name match among
all competing S1s by > 0.05" gives 93% precision / 52% recall, against the
model's 87% / 35%.

## Plan to > 0.99

| Cycle | What | State |
|---|---|---|
| **1** | Competition features (rank and margin of this S1 among all S1s competing for the candidate, by name trigram / token-set / embedding, plus rival count); `street_core_ratio` (street name without number/type; targets France + house-number FPs); `name_acronym_match` | ✅ code + tests; ▶ run: `python run_union_pipeline.py --force-features --run dev_v11` |
| 2 | Stacking: 2-fold out-of-fold model scores → competition by the model's own score; cross-encoder (from e5) on borderline pairs as one more feature | next |
| – | Last ~5 h: `Documentation_template.md`, README, submission zip, final validation | reserve |

Each cycle is gated: dev_val must improve for India **and** US before it's submitted.

## Key artifacts (all under `data/`, gitignored — copy the folder to move machines)

| Path | What |
|---|---|
| `processed/candidates/<split>/union_retriever_k20.parquet` | Token + embedding candidates (current) |
| `processed/features/<split>_union_retriever_k20/` | 48 features per pair (current) |
| `processed/models/retriever/` | Fine-tuned e5-small retriever (triplet acc 0.923 → 0.998) |
| `processed/models/dev_union/`, `final_union/` | v1.0.0 dev / final XGBoost |
| `processed/errors/dev_union_errors.{parquet,tsv}` | Every wrong dev_val decision with texts |
| `processed/finetune/` | Retriever fine-tuning pairs (fit entities only) |

## Tools

- `python src/run_union_pipeline.py` — embed → union → features → dev model, with a whole-pipeline progress bar.
- `python src/error_analysis.py --run <dev run>` — error file + where the F0.5 is lost.
- `python tools/debug_api.py` → http://127.0.0.1:8000/docs — inspect any entity or pair through every step.

## Rules checklist

- ✅ XGBoost/LightGBM; multilingual-e5-small (MIT, 118M); fine-tuning on train data
  only; hand-written normalization lists; unlabeled test text.
- ❌ Hosted APIs (LLM/embedding/geocoding/registries), internet datasets, models
  >8B or non-MIT/Apache; never upload competition data anywhere.
- `country` is an open set everywhere; France takes the same code path.

## Gotchas

- The PyPI LightGBM has no GPU support, so `--gpu` uses XGBoost CUDA.
- Output TSVs must use `\n` line endings (enforced in `export.py`, and tested).
- Full training needs ~30–40 GB RAM; union features ~10 GB per split on disk.
- Run long jobs in a plain terminal (no `| Tee-Object`) so progress bars redraw in place.
- `predict.py --features` must match the features the model was trained on (checked).
