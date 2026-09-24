# Qbyte — Business Entity Resolution

ML Challenge 2026 entry: matching business entity records across 3 noisy
data sources (blocking + pairwise matching). Full problem spec, scoring, and
submission rules are in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

## Status

Project scaffolding only. Dataset is in place (`data/raw/train`,
`data/raw/test`) but no blocking, matching, or pipeline logic has been
implemented yet.

## Structure

- `data/` — raw and processed data (gitignored, see [data/README.md](data/README.md))
- `business_entity_resolution/` — the submittable package (`src/`, `README.md`, `requirements.txt`)
- `notebooks/` — exploration notebooks
- `tests/` — unit tests
- `pipelines/` — scripts to run the pipeline and build the submission zip
- `configs/` — run configuration
- `output/` — generated `matching_results.tsv` / `candidate_pairs.tsv` (gitignored)
- `utils/validate_submission.py` — official validator, run before every submission

See [ARCHITECTURE.md](ARCHITECTURE.md) for the planned pipeline design and
[Documentation_template.md](Documentation_template.md) for the submission
writeup template.

## Setup

```bash
pip install -r business_entity_resolution/requirements.txt
```
