# Data

- `raw/train/` — `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
- `raw/test/` — `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` (no ground truth)
- `processed/` — cleaned/derived data (gitignored)

All files are `.tsv` (read with `sep="\t"`). Each source file has
`entity_id` (prefixed `S1-`/`S2-`/`S3-`), `business_name`, `business_address`,
`country`. `train_ground_truth.tsv` has `source1_entity_id` and
`matched_entity_ids` (comma-separated `S2-`/`S3-` IDs, empty for singletons).

Full spec — noise patterns, output format, scoring — is in
[../PROBLEM_STATEMENT.md](../PROBLEM_STATEMENT.md).

No data is committed to this repository (`*.tsv` is gitignored). The
canonical copy lives in S3 (bucket/prefix/version pinned in
[../configs/data_source.yaml](../configs/data_source.yaml)); pull it with:

```bash
aws s3 sync s3://ml-challenge-qbyte/entity-resolution/raw/extracted/student_resource/dataset/train/ data/raw/train/
aws s3 sync s3://ml-challenge-qbyte/entity-resolution/raw/extracted/student_resource/dataset/test/  data/raw/test/
```
