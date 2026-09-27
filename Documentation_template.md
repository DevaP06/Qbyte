# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Qbyte  
**Team Members:** Devashish, Shekhar Deshmukh, Tejas Tandon  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A four-stage cascade: scalable candidate generation (country-partitioned IDF token
blocking ∪ exact kNN over a fine-tuned multilingual-e5-small retriever), a 57-feature
XGBoost pair classifier, three rounds of cross-encoders plus a multilingual-e5-base
cross-encoder on the pairs the classifier cannot settle, and a level-2 XGBoost stacker
that learns when to trust which score. Key innovations: *competition features* (how this
S1 ranks against every other S1 claiming the same record) and cross-encoder training
that reaches France (no labels) through unlabeled agreement pairs and synthetic
French hard negatives. Dev macro F0.5 0.98921 (India 0.99000 / US 0.98868), portal 0.985226.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Noise is mostly character- and script-level:** typos (`kd-fottcabre`), merged/split
  words (`mubarakpolymers`), reordered tokens, truncated addresses, and ~15% non-Latin
  India names (Devanagari, Kannada). Token blocking missed 27% of non-Latin India matches.
- **No postal codes and no stable address schema:** US addresses have 2–3 comma
  components, India 3–8+ with landmarks. We parse positionally (head / middle bag / tail)
  instead of guessing a schema.
- **Name-only records** (no address) are candidates of ~37 S1 entities each. Whether one
  belongs to *this* S1 depends on the rivals, which no pair-local feature can see. This
  was the single largest error source (§5).
- **Many-to-one structure:** 0 of 7,638,365 training pairs have an S2/S3 id claimed by two
  S1 entities, while an S1 has ~3.5 matches on average, so an S2/S3 record goes to at most
  one S1.
- **France has no training labels.** Its errors are text judgments: the same address
  with one business word substituted (`lutins institut sas` vs `lutins comite sas`),
  another street at the same house number. The legal form is part of a French business's
  identity: 17.6% of distinct French S1 businesses share their name and city with another
  one, and they differ only by legal form (`3eme amicale sarl` vs `3eme amicale eurl`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier cascade (hybrid: lexical + dense retrieval, GBDT, transformer cross-encoders, stacking)  
**Core Innovation:** Candidate-side competition features, plus a cross-encoder curriculum
that transfers to an unseen country (France) without its labels. `country` is an open set
everywhere: every stage groups by whatever string is in `country`, and France takes the
identical code path.

Leakage control: train S1 entities are split once (seed 42) into fit / early-stop /
dev_val. The retriever, GBDT and cross-encoders see only fit entities (asserted in code),
and the stacker is trained and 2-fold cross-validated on dev_val. Every change was gated
on dev CV not dropping for India and US.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. *Token blocking:* an inverted index per country over normalized name tokens and
     address tokens. Generic/legal tokens are excluded as keys and the rest weighted by
     IDF. Each S1 keeps its top 40 records by IDF-weighted overlap. Name-only keys cap
     India at 74.9% recall, so address tokens are needed.
  2. *Dense retrieval:* multilingual-e5-small (MIT, 118M) fine-tuned on (S1, true match,
     hard negative) triplets from fit entities (triplet accuracy 0.923 → 0.998). Exact
     GPU kNN within each country, top 20 per S1.
  3. The union of both. The embedding cosine is computed for every pair.
- **Candidate pairs generated:** 97,191,350 on test (56.1 per S1 entity, 1,732,544 S1
  entities, 0 with no candidates). Against the full S1 × (S2 ∪ S3) cross product
  (1.73M × 9.97M = 1.7 × 10¹³) this is a **reduction ratio of 99.99944%**. Train:
  124,112,932 pairs (56.2 per S1).
- **How you ensured true matches were not lost:** recall was measured against the ground
  truth at every stage. Token blocking alone reaches 87.1% (India) / 94.7% (US) pair
  recall at top 40, and raising it to 300 added only +4.4 / +1.6 points. Dense retrieval
  targets exactly the misses (typos, merged words, cross-script). The union's macro F0.5
  ceiling on dev_val is **0.9992** (India 0.9993 / US 0.9991), i.e. under 0.1% of the
  score is lost to blocking. It scales: it is an inverted index plus batched matrix
  products per country, never all-pairs, and cost grows linearly with records × k.
- **Cascade after blocking:** the GBDT passes only pairs scoring ≥ 0.02 to the
  cross-encoders and stacker: 7,377,136 test pairs (4.3 per S1). `candidate_pairs.tsv`
  lists the full 97.2M set the GBDT scores.

---

## 4. Matching Model

**Features used (57, all country-agnostic; IDF computed per country):**
- Name features: rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on
  Unicode-preserving text, ASCII-folded token-set, token Jaccard, token IDF cosine and
  containment (both directions), char-trigram Dice and IDF cosine, digit match/conflict,
  lengths and ratio, non-ASCII fraction, acronym match.
- Address features: full / head / middle-bag / tail similarities, token Jaccard, IDF
  cosine and containment, digit (house number) match / conflict / Jaccard, both-present
  flag, length ratio, street-core ratio (street name without number and type).
- Other: embedding cosine and rank, blocking score/rank, gaps and ranks of the S1's
  candidates on key similarities, source flag, and **7 competition features**: this S1's
  rank and margin against all other S1s sharing the candidate, by name trigram, name
  token-set and embedding, plus the rival count.

**Model type:**
1. XGBoost (CUDA, 127-leaf lossguide, 1,569 rounds) on 99.3M fit pairs.
2. Cross-encoders on pairs with GBDT ≥ 0.02, each reading both records together:
   - round 1: from the fine-tuned retriever, on 2.98M fit pairs
   - round 2: continued on 2.17M pairs (new fit entities + French *agreement pairs*,
     i.e. unlabeled test pairs where the GBDT and round 1 agree with high confidence)
   - round 3: continued on 1.76M pairs (replay + agreement + synthetic French hard
     negatives made by swapping the business word or street, synthetic positives,
     French S1–S1 negatives)
   - multilingual-e5-base (MIT, 278M): trained on rounds 1–3 data (35% French), scoring
     only the 1.94M pairs that the GBDT and round 3 do not both call confident matches.
3. A level-2 XGBoost stacker over the GBDT, round-2, round-3 and e5-base logits, the
   S1-local context (ranks, gaps, number of scored candidates, model disagreement),
   14 pair features, and sibling coherence (embedding cosine between the candidate and
   the S1's strongest other candidate). It is bagged over 5 seeds and has no country input.

**Threshold selection method:** direct macro F0.5 maximization, averaged over every S1
entity exactly as the challenge defines it: GBDT 0.72 on dev_val, stacker 0.70 on
out-of-fold dev_val scores. Post-processing gives each S2/S3 id to at most one S1 (its
highest-scoring one), which measured +0.0004.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.98921 on dev_val (2-fold CV by entity; India 0.99000, US
  0.98868). Portal 0.985226. The gap is France (~15% of test, implied F0.5 ≈ 0.955).
- **Common false positives (wrong merges):** near-duplicate distractors (same brand,
  another branch), same address but a different business (shared buildings, one
  business word substituted), house-number conflicts on the same street, and name-only
  records that fit several S1s.
- **Common false negatives (missed matches):** name-only records with no address (the
  largest loss before competition features), heavy typos / truncated addresses, DBA or
  trade names that differ from the legal name, and non-Latin names against Latin ones.

Loss by cause at v1.0.0, measured by flipping each cause's wrong decisions and
rescoring dev_val:

| Cause | Gain if fixed |
|---|---|
| FN: name-only candidate | +0.0087 |
| FN: typos / suffix / partial address | +0.0030 |
| FP: name-only candidate | +0.0019 |
| FN: different / DBA trade name | +0.0017 |
| FP: near-duplicate distractor | +0.0017 |
| FP: same address, different business | +0.0012 |
| FP: house-number conflict | +0.0009 |
| FN: non-Latin name | +0.0004 |

Competition features targeted the top row (+0.0058 dev). The cross-encoders and
stacker targeted the text-level errors.

---

## 6. Conclusion

Most of the score comes from recall-safe candidate generation (ceiling 0.9992 at
56 candidates per S1) and from giving the matcher context a single pair cannot
provide: rival S1s, sibling matches, and a model that reads both records together.
Lesson for the unseen country: add signal, never delete it. Stripping French legal
forms looked like normalization but merged distinct businesses and cost 0.0036 on the portal.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds all source in `src/`, plus `README.md` with the
full command sequence and `requirements.txt` with pinned versions. Entry points, run from
`src/`:

| Stage | Command |
|---|---|
| Cleaning | `clean_all.py` (`normalize.py`, `address_parser.py`, `io_utils.py`, `resources/legal_suffixes.json`) |
| Token blocking | `blocking_token.py --split train/test` |
| Retriever | `finetune_data.py`, `finetune_embeddings.py`, `embeddings.py` |
| Union + features + GBDT | `run_union_pipeline.py` (`candidate_generation.py`, `features.py`, `competition_features.py`, `train_classifier.py`), `predict.py` |
| Cross-encoders | `cross_encoder.py` (round 1), `run_v22.py` (round 2), `run_v23.py` (round 3), `run_v25.py` / `ce_base.py` (e5-base) |
| Final output | `stack2.py apply --bag 5 --coherence --tag _r2,_r3 --extra-tag _base` → `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

### B. Additional Results

| Version | Change | dev F0.5 | Portal |
|---|---|---|---|
| v0.1 | token blocking + 43 features + XGBoost | 0.9469 | 0.9378 |
| v1.0 | + fine-tuned embedding retrieval, embedding features | 0.9799 | 0.9724 |
| v1.1 | + competition, street-core, acronym features | 0.9857 | 0.9780 |
| v2.0 | + cross-encoder, linear blend | 0.9872 | 0.9812 |
| v2.2 | + CE round 2 (French agreement pairs), level-2 stacker | 0.9888 | 0.9835 |
| v2.3 | + CE round 3 (synthetic French negatives), 5-seed bagging, coherence | 0.98909 | 0.9849 |
| **v2.5** | + multilingual-e5-base cross-encoder | **0.98921** | **0.985226** |

Measured and rejected: bipartite (1:1) assignment (+0.0000), label-free per-country
threshold (−0.0015 simulated with US as the unseen country), pseudo-labeled GBDT
(+0.0008), reverse retrieval (≤ +0.0002), larger retrieval k (≤ +0.0009 for 2× the
pairs), stacker over all 57 features (+0.0001), and French normalization that strips
legal forms (portal −0.0036).
