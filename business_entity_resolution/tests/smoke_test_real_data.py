"""Real-data smoke test for the cleaning stage (not a unit test).

Reads a manageable sample from each real source/test file (never the full
multi-million-row files), runs clean_dataframe, prints targeted before/after
examples, and verifies the data-integrity invariants: row count preserved,
IDs unchanged, country values unchanged.

Run: python business_entity_resolution/tests/smoke_test_real_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "business_entity_resolution" / "src"))

import config  # noqa: E402
from io_utils import clean_dataframe, read_sample  # noqa: E402

SAMPLE_N = 20_000


def check_integrity(raw_df, clean_df, label):
    assert len(raw_df) == len(clean_df), f"{label}: row count changed"
    assert (raw_df[config.ENTITY_ID_COL].to_numpy() == clean_df[config.ENTITY_ID_COL].to_numpy()).all(), (
        f"{label}: entity_id changed"
    )
    assert (raw_df[config.COUNTRY_COL].fillna("").to_numpy() == clean_df[config.COUNTRY_COL].to_numpy()).all(), (
        f"{label}: country value changed"
    )
    print(f"  [{label}] integrity OK: {len(raw_df)} rows, IDs and countries unchanged")


def show_examples(raw_df, clean_df, label, predicate, n=4):
    mask = raw_df[config.NAME_COL].fillna("").map(predicate) | raw_df[config.ADDRESS_COL].fillna("").map(predicate)
    idx = raw_df.index[mask][:n]
    if len(idx) == 0:
        print(f"  [{label}] (no matching examples in this sample)")
        return
    print(f"  [{label}]")
    for i in idx:
        print(f"    raw name:    {raw_df.loc[i, config.NAME_COL]!r}")
        print(f"    raw address: {raw_df.loc[i, config.ADDRESS_COL]!r}")
        print(f"    -> name_original_normalized: {clean_df.loc[i, 'name_original_normalized']!r}")
        print(f"    -> name_ascii_folded:        {clean_df.loc[i, 'name_ascii_folded']!r}")
        print(f"    -> name_tokens:              {clean_df.loc[i, 'name_tokens']!r}")
        print(f"    -> address_present/head/mid/tail: {clean_df.loc[i, 'address_present']}"
              f" / {clean_df.loc[i, 'address_head']!r} / {clean_df.loc[i, 'address_middle']!r}"
              f" / {clean_df.loc[i, 'address_tail']!r}")
        print()


def run_for_file(path, label):
    print(f"=== {label}: {path} ===")
    raw_df = read_sample(path, SAMPLE_N)
    clean_df = clean_dataframe(raw_df)
    check_integrity(raw_df, clean_df, label)

    has_non_ascii = lambda s: any(ord(ch) > 127 for ch in s)
    has_legal_suffix = lambda s: any(
        w in s.lower() for w in ("pvt", "private", "ltd", "limited", "corp", "inc", "llc", "llp")
    )
    has_domain = lambda s: ".com" in s.lower() or ".in" in s.lower() or ".net" in s.lower() or "www." in s.lower()
    has_digit = lambda s: any(ch.isdigit() for ch in s)

    show_examples(raw_df, clean_df, "non-ASCII script names", has_non_ascii)
    show_examples(raw_df, clean_df, "legal-suffix names", has_legal_suffix)
    show_examples(raw_df, clean_df, "domain-style names", has_domain)
    show_examples(raw_df, clean_df, "names/addresses with digits", has_digit)

    empty_mask = raw_df[config.ADDRESS_COL].fillna("").map(lambda s: s.strip() == "")
    n_empty = int(empty_mask.sum())
    print(f"  [empty addresses] {n_empty}/{len(raw_df)} in this sample "
          f"({100.0 * n_empty / len(raw_df):.2f}%)")
    if n_empty:
        i = raw_df.index[empty_mask][0]
        print(f"    raw address: {raw_df.loc[i, config.ADDRESS_COL]!r}")
        print(f"    -> address_present: {clean_df.loc[i, 'address_present']}, "
              f"head: {clean_df.loc[i, 'address_head']!r}, tail: {clean_df.loc[i, 'address_tail']!r}")
    print()


def main():
    files = [
        (config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["source1"], "train_source1"),
        (config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["source2"], "train_source2"),
        (config.DATA_RAW_TRAIN_DIR / config.TRAIN_FILES["source3"], "train_source3"),
        (config.DATA_RAW_TEST_DIR / config.TEST_FILES["source1"], "test_source1"),
    ]
    for path, label in files:
        run_for_file(path, label)
    print("ok")


if __name__ == "__main__":
    main()
