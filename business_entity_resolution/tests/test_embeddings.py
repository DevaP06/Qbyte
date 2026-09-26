"""Unit tests for src/embeddings.py (CPU only; the encoder is a stub).

Run: python business_entity_resolution/tests/test_embeddings.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from candidate_generation import merge_candidate_keys, rowwise_cosine  # noqa: E402
from embeddings import embed_texts, knn_by_country, model_key  # noqa: E402


def _normalized(n, d, seed):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def test_knn_matches_brute_force_and_never_crosses_countries():
    emb = _normalized(60, 8, seed=0)
    countries = np.array(["US"] * 25 + ["India"] * 20 + ["France"] * 15)
    is_s1 = np.zeros(60, dtype=bool)
    is_s1[[0, 1, 2, 25, 26, 45]] = True
    # block_bytes tiny -> one query per batch, exercising the batching loop
    q, t, s = knn_by_country(emb, countries, is_s1, k=3, device="cpu", block_bytes=1)
    assert (countries[q] == countries[t]).all(), "a neighbour crossed a country boundary"
    assert not is_s1[t].any(), "an S1 record was returned as a candidate"
    full = emb.astype(np.float32) @ emb.astype(np.float32).T
    for row in np.flatnonzero(is_s1):
        targets = np.flatnonzero(~is_s1 & (countries == countries[row]))
        expected = targets[np.argsort(-full[row, targets], kind="stable")[:3]]
        got = t[q == row]
        assert list(got) == list(expected), (row, got, expected)
        assert np.all(np.diff(s[q == row]) <= 1e-6), "scores must be descending"


def test_knn_k_larger_than_targets():
    emb = _normalized(5, 4, seed=1)
    countries = np.array(["France"] * 5)
    is_s1 = np.array([True, False, False, True, False])
    q, t, _ = knn_by_country(emb, countries, is_s1, k=10, device="cpu")
    assert len(t[q == 0]) == 3  # only 3 non-S1 records exist


class _StubEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.calls.append(list(texts))
        rows = [np.random.default_rng(abs(hash(t)) % 2**32).normal(size=4) for t in texts]
        x = np.array(rows, dtype=np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_embed_texts_dedups_prefixes_and_fills_in_order():
    texts = np.array(["a, x", "b, y", "a, x", "c, z"], dtype=object)
    out = np.zeros((4, 4), dtype=np.float16)
    enc = _StubEncoder()
    embed_texts(texts, enc, out, prefix="query: ", chunk=2)
    encoded = [t for call in enc.calls for t in call]
    assert encoded == ["query: a, x", "query: b, y", "query: c, z"]  # each unique text once, prefixed
    assert np.array_equal(out[0], out[2])  # duplicate text -> identical row
    assert np.allclose(np.linalg.norm(out.astype(np.float32), axis=1), 1.0, atol=1e-2)


def test_merge_candidate_keys():
    token_keys, token_score = np.array([10, 20, 30]), np.array([5.0, 4.0, 3.0], dtype=np.float32)
    emb_keys, emb_rank = np.array([30, 40]), np.array([1, 0], dtype=np.float32)
    keys, score, rank = merge_candidate_keys(token_keys, token_score, emb_keys, emb_rank)
    assert list(keys) == [10, 20, 30, 40]
    assert list(score) == [5.0, 4.0, 3.0, 0.0]  # 40: embedding-only -> token score 0
    assert np.isnan(rank[0]) and np.isnan(rank[1]) and rank[2] == 1 and rank[3] == 0


def test_rowwise_cosine():
    emb = _normalized(10, 6, seed=2)
    q, t = np.array([0, 3, 7]), np.array([1, 3, 2])
    got = rowwise_cosine(emb, q, t, batch=2)
    e = emb.astype(np.float32)
    assert np.allclose(got, (e[q] * e[t]).sum(axis=1), atol=1e-5)
    assert np.isclose(got[1], 1.0, atol=1e-2)  # a row with itself


def test_model_key():
    assert model_key("intfloat/multilingual-e5-small") == "intfloat__multilingual-e5-small"
    assert model_key(str(ROOT)) == ROOT.name  # existing dir -> its name


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
