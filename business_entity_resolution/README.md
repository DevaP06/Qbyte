# Business Entity Resolution

Source package for the entity resolution solution (blocking + matching).

## Status

**Implemented:** the cleaning/normalization stage — `src/normalize.py`,
`src/address_parser.py`, `src/io_utils.py`, `src/config.py`,
`src/resources/legal_suffixes.json`. See `../plan.md` for the full pipeline
plan; this stage covers plan step 1 only (build order, "Today").

**Not yet implemented:** token blocking, embeddings/ANN, feature engineering,
classification, thresholding, post-processing/export. Nothing here writes
`output/candidate_pairs.tsv` or `output/matching_results.tsv` yet.

## Usage

```bash
pip install -r requirements.txt
```

Run the unit tests (stdlib `assert`-based, no pytest dependency — same
convention as `../tests/test_create_submission.py`):

```bash
python business_entity_resolution/tests/test_normalize.py
python business_entity_resolution/tests/test_address_parser.py
```

Run the real-data smoke test (reads a 20K-row sample per file, never the
full multi-million-row files; prints before/after examples and checks the
row-count/ID/country integrity invariants):

```bash
python business_entity_resolution/tests/smoke_test_real_data.py
```

## Cleaning stage API

- `normalize.normalize_text(raw) -> NormalizedText` — `raw`,
  `original_normalized` (Unicode-preserving), `ascii_folded` (Latin-only,
  kept separate, never replaces the original), `tokens` (legal-suffix
  variants canonicalized, nothing deleted), `digit_tokens`.
- `normalize.is_blocking_token(token)` / `normalize.blocking_tokens(tokens)`
  — identifies generic/legal tokens (pvt, ltd, services, india, ...) as
  unsuitable *blocking keys* without removing them from the token list.
- `normalize.normalize_country(raw)` — whitespace-trim only; identical code
  path for every country string, including one never seen before (France).
- `address_parser.parse_address(raw) -> AddressComponents` — heuristic
  comma-split into `head` (street-like)/`middle` (bag)/`tail` (region-like),
  degrading gracefully for 0/1/2/3/8+ components and empty input.
- `io_utils.clean_dataframe(df)` — applies the above to one chunk of a
  source file; `io_utils.clean_source_file_chunks(path)` does it for a full
  file in bounded-memory chunks; `io_utils.cache_or_build(...)` is a
  minimal Parquet cache-or-recompute helper.
