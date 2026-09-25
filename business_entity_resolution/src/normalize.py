"""Deterministic text normalization and tokenization for business names.

Stdlib-only (unicodedata, re, json) so this module is testable and reusable
without pulling in pandas/etc. Produces multiple, non-destructive text
representations per the project plan:

    raw                 -> the untouched input (converted to "" for None/NaN)
    original_normalized -> NFKC + lowercase + domain-noise-stripped +
                            leading-punctuation-stripped, Unicode-PRESERVING
                            (kept for the future embedding stage; ~15% of
                            sampled India names are non-Latin script and must
                            not be destroyed)
    ascii_folded        -> original_normalized with diacritics stripped and
                            non-ASCII characters dropped, for classical
                            string-similarity features on Latin text only.
                            Never used as a replacement for original_normalized.
    tokens              -> deterministic tuple of tokens derived from
                            original_normalized, with legal-suffix variants
                            CANONICALIZED (not deleted) via legal_suffixes.json
    digit_tokens        -> frozenset of tokens that are standalone digit runs
                            (e.g. {"100"} from "Suite 100"), for later
                            "Suite 100 vs Suite 101" conflict features.

Legal/generic tokens (pvt, ltd, inc, services, india, ...) are NEVER deleted
from the token list here -- they remain available as features. This module
only exposes `is_blocking_token`/`blocking_tokens` so a later blocking stage
can filter them out as blocking *keys* without losing them as data.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

_RESOURCES_PATH = Path(__file__).resolve().parent / "resources" / "legal_suffixes.json"


def _load_suffix_resources(path: Path) -> tuple[dict, frozenset]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    suffix_map = {k.lower(): v.lower() for k, v in data["legal_suffix_canonical_map"].items()}
    stopwords = frozenset(w.lower() for w in data["generic_blocking_stopwords"])
    return suffix_map, stopwords


LEGAL_SUFFIX_CANONICAL_MAP, GENERIC_BLOCKING_STOPWORDS = _load_suffix_resources(_RESOURCES_PATH)

# Domain/URL noise, applied in this order:
#   1. drop a literal "www." prefix wherever it occurs
#   2. drop a recognized TLD suffix glued onto a token (".com", ".net", ...)
# Only the "www." / ".tld" fragments are removed -- never the whole token --
# so "wilfordhancock.com" -> "wilfordhancock" (the likely real name) instead
# of being deleted outright.
_WWW_RE = re.compile(r"\bwww\.")
_TLD_SUFFIX_RE = re.compile(r"(?<=[a-z0-9])\.(?:com|net|org|co|io|biz|info|in)\b")

# Leading punctuation/noise (e.g. "-- Holloway Peak Inc"): strip any run of
# whitespace/ASCII punctuation from the start of the string. Matched against
# an explicit ASCII punctuation set rather than `[^\w]` -- `\w`'s complement
# also matches Unicode combining marks (category Mn/Mc), which can
# legitimately open a grapheme cluster in some Indic-script renderings; using
# `\w` here risked silently trimming a leading mark off a non-Latin name.
_LEADING_NOISE_RE = re.compile(r"^[\s" + re.escape(string.punctuation) + r"]+")

_WHITESPACE_RE = re.compile(r"\s+")

# Tokenizer: split on whitespace and ASCII punctuation, keep everything else.
#
# `\w+` was tried first and rejected: Python's `\w` only matches Unicode
# categories L*/N*/_, which EXCLUDES combining marks (category Mn/Mc). Most
# Indic scripts (Devanagari, Kannada, ...) spell a syllable as a base
# consonant plus a combining vowel sign (matra); splitting on `\w` shatters
# each word into its individual consonants and silently drops every matra
# ("राम" -> "र", "म" instead of staying "राम"). Splitting on an explicit
# separator set (whitespace + ASCII punctuation) instead treats combining
# marks as ordinary content, so a word stays one token regardless of script.
# Apostrophes/slashes/brackets/pipes still act as separators, matching the
# previous behavior for Latin text (e.g. "orelee's" -> "orelee", "s").
_SEPARATOR_CHARS = string.punctuation.replace("_", "")
_SEPARATOR_RE = re.compile(r"[\s" + re.escape(_SEPARATOR_CHARS) + r"]+", re.UNICODE)


def _tokenize(s: str) -> Tuple[str, ...]:
    return tuple(tok for tok in _SEPARATOR_RE.split(s) if tok)


def to_clean_str(raw: Optional[object]) -> str:
    """Convert a raw cell value to a string, treating None/NaN/whitespace as "".

    Handles the three blank forms the data actually contains: ``None``,
    float ``NaN`` (checked via ``x != x``, true only for NaN, so this needs no
    numpy/pandas import), and empty/whitespace-only strings.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and raw != raw:  # NaN
        return ""
    s = str(raw).strip()
    return s


def is_blank(raw: Optional[object]) -> bool:
    """True for None/NaN/""/whitespace-only values."""
    return to_clean_str(raw) == ""


def _strip_domain_noise(s: str) -> str:
    s = _WWW_RE.sub("", s)
    s = _TLD_SUFFIX_RE.sub("", s)
    return s


def _collapse_whitespace(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()


def _ascii_fold(s: str) -> str:
    """Strip diacritics from Latin text; drop non-ASCII characters entirely.

    Non-Latin scripts (Devanagari, Kannada, ...) have no ASCII decomposition
    and are simply removed from this representation -- by design, since it
    exists only for classical string features on Latin text. The
    Unicode-preserving `original_normalized` string is what carries that
    information forward for embedding.
    """
    decomposed = unicodedata.normalize("NFKD", s)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_bytes = without_marks.encode("ascii", "ignore")
    return _collapse_whitespace(ascii_bytes.decode("ascii"))


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    original_normalized: str
    ascii_folded: str
    tokens: Tuple[str, ...]
    digit_tokens: FrozenSet[str]


def normalize_text(raw: Optional[object]) -> NormalizedText:
    """Run the full normalization pipeline on one text field (name or similar).

    Deterministic and side-effect-free. Safe on None/NaN/empty input.
    """
    raw_str = to_clean_str(raw)

    nfkc = unicodedata.normalize("NFKC", raw_str)
    lowered = nfkc.lower()
    domain_stripped = _strip_domain_noise(lowered)
    collapsed = _collapse_whitespace(domain_stripped)
    original_normalized = _LEADING_NOISE_RE.sub("", collapsed)

    ascii_folded = _ascii_fold(original_normalized)

    raw_tokens = _tokenize(original_normalized)
    tokens = tuple(LEGAL_SUFFIX_CANONICAL_MAP.get(tok, tok) for tok in raw_tokens)
    digit_tokens = frozenset(tok for tok in tokens if tok.isdigit())

    return NormalizedText(
        raw=raw_str,
        original_normalized=original_normalized,
        ascii_folded=ascii_folded,
        tokens=tokens,
        digit_tokens=digit_tokens,
    )


def is_blocking_token(token: str) -> bool:
    """False for generic/legal tokens (pvt, ltd, inc, services, india, ...).

    These tokens are poor *blocking keys* (carry no discriminative signal at
    retrieval time) but are NOT deleted from `NormalizedText.tokens` -- they
    remain available for downstream features (e.g. IDF-weighted overlap).
    """
    return token.lower() not in GENERIC_BLOCKING_STOPWORDS


def blocking_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    """Subset of `tokens` suitable as inverted-index blocking keys."""
    return tuple(tok for tok in tokens if is_blocking_token(tok))


def normalize_country(raw: Optional[object]) -> str:
    """Whitespace-trim only. No case-folding, no country-specific lookup.

    Every country string (including "France", unseen at train time) goes
    through this identical path -- there is no `if country == ...` branch
    anywhere in this pipeline.
    """
    return to_clean_str(raw)
