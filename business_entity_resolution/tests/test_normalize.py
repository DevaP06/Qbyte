"""Unit tests for src/normalize.py. Run: python business_entity_resolution/tests/test_normalize.py"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from normalize import (  # noqa: E402
    blocking_tokens,
    is_blocking_token,
    normalize_country,
    normalize_text,
)


def test_unicode_preserved():
    # Devanagari: must survive into original_normalized untouched (~15% of
    # sampled India names are non-Latin script -- this is the core
    # non-destructiveness guarantee).
    n = normalize_text("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert "राम" in n.original_normalized
    assert "मार्केटिंग" in n.original_normalized
    assert n.tokens[0] == "राम"

    # Kannada.
    n = normalize_text("ಶ್ರೀ ಟ್ರೇಡರ್ಸ್")
    assert "ಶ್ರೀ" in n.original_normalized
    assert len(n.tokens) == 2

    # Mixed Latin + Devanagari in one field (seen in real Source-3 addresses).
    n = normalize_text("M/s Basera Advisors उत्तर प्रदेश")
    assert "basera" in n.original_normalized
    assert "उत्तर" in n.original_normalized


def test_case_insensitivity():
    variants = ["ABC Company", "abc company", "AbC CoMpAnY"]
    normalized = {normalize_text(v).original_normalized for v in variants}
    assert normalized == {"abc company"}


def test_nfkc_compatibility_normalization():
    # Full-width Latin (compatibility) characters collapse to plain ASCII
    # letters under NFKC, matching the plan's "full-width Unicode -> NFKC"
    # example.
    fullwidth = "Ａｂｃ"  # fullwidth "Abc"
    n = normalize_text(fullwidth)
    assert n.original_normalized == "abc"


def test_domain_noise():
    # Whole name is a slugified domain -- strip only the TLD, keep the name.
    assert normalize_text("wilfordhancock.com").original_normalized == "wilfordhancock"
    assert normalize_text("heassociates.com").original_normalized == "heassociates"
    assert normalize_text("7m.com").original_normalized == "7m"

    # Domain glued on with "www." and a pipe separator alongside a real name:
    # the readable name must survive, and neither "www" nor the TLD remains
    # as a token.
    n = normalize_text("SHIVSHAKTI OVERSEAS CORPORATION | www.shivshakti.com")
    assert "shivshakti overseas corporation" in n.original_normalized
    assert "www" not in n.tokens
    assert "com" not in n.tokens  # the bare TLD token itself must not remain
    assert "shivshakti" in n.tokens  # the domain's meaningful part is preserved


def test_legal_suffix_canonicalization_consistency():
    pairs = [
        ("ABC Pvt Ltd", "ABC Private Limited"),
        ("ABC Corp", "ABC Corporation"),
    ]
    for a, b in pairs:
        assert normalize_text(a).tokens == normalize_text(b).tokens, (a, b)

    assert normalize_text("ABC Inc").tokens == ("abc", "inc")
    assert normalize_text("ABC LLC").tokens == ("abc", "llc")
    assert normalize_text("ABC LLP").tokens == ("abc", "llp")


def test_legal_suffix_conservative_no_false_positives():
    # Ordinary words that merely resemble a suffix must not be mangled --
    # canonicalization only matches whole tokens, never substrings.
    n = normalize_text("Corporate Bakery")
    assert n.tokens == ("corporate", "bakery")

    # A domain-derived token that happens to contain "private" as a
    # substring must not be treated as the "private" suffix.
    n = normalize_text("prprivate.com")
    assert n.tokens == ("prprivate",)


def test_generic_tokens_kept_but_excluded_from_blocking():
    n = normalize_text("Global Services India Pvt Ltd")
    assert set(n.tokens) == {"global", "services", "india", "pvt", "ltd"}
    assert not is_blocking_token("pvt")
    assert not is_blocking_token("services")
    assert not is_blocking_token("india")
    assert not is_blocking_token("ltd")
    assert is_blocking_token("global")
    assert blocking_tokens(n.tokens) == ("global",)


def test_digit_token_extraction():
    assert normalize_text("Suite 100").digit_tokens == frozenset({"100"})
    assert normalize_text("Suite 101").digit_tokens == frozenset({"101"})
    assert normalize_text("Suite 100").digit_tokens.isdisjoint(
        normalize_text("Suite 101").digit_tokens
    )
    assert normalize_text("Building 42").digit_tokens == frozenset({"42"})
    # "42-B" tokenizes to ("42", "b"); "b" is not a standalone digit token.
    n = normalize_text("42-B")
    assert n.tokens == ("42", "b")
    assert n.digit_tokens == frozenset({"42"})
    # A digit embedded in an alphanumeric token ("7m") is not a standalone
    # digit token -- digits are only extracted from all-digit tokens.
    assert normalize_text("7m").digit_tokens == frozenset()


def test_null_handling():
    for blank in (None, float("nan"), "", "   "):
        n = normalize_text(blank)
        assert n.original_normalized == ""
        assert n.ascii_folded == ""
        assert n.tokens == ()
        assert n.digit_tokens == frozenset()
    assert math.isnan(float("nan"))  # sanity check on the NaN literal used above


def test_country_independence():
    # Same function, no per-country branch, for every observed country
    # string -- including France, unseen at train time.
    for country in ("US", "India", "France", "  US  "):
        result = normalize_country(country)
        assert result == country.strip()


def test_ascii_folded_kept_separate_from_original():
    n = normalize_text("Café Jáckson")
    assert n.original_normalized == "café jáckson"  # Unicode preserved
    assert n.ascii_folded == "cafe jackson"  # separate, folded representation
    assert n.original_normalized != n.ascii_folded


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
