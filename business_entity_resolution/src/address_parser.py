"""Heuristic, comma-based address decomposition.

The plan explicitly rejects full tree-edit distance for addresses: there is
no stable schema (US addresses run ~2-3 comma-separated components; Indian
addresses commonly run 3-8+, with landmarks, localities, and reordered
components; postal codes are absent from the corpus entirely). Building a
"tree" would itself just be a guess.

Instead this module does a positional comma-split into three pieces:

    head   ~= street-like beginning        (first component)
    tail   ~= region-like ending           (last component)
    middle ~= everything else, as a bag    (components strictly in between)

This makes no assumption about component count and degrades gracefully for
0, 1, 2, 3, or 8+ components. No geocoding, no postal-code database, no
country-specific branching -- the same code path runs for every country
string.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

from normalize import to_clean_str

_WHITESPACE_RE = re.compile(r"\s+")
# Trim stray punctuation left dangling at a component's edges after
# splitting on commas (e.g. a lone leading "-" or trailing "."). Matched
# against an explicit ASCII punctuation set rather than `[^\w]` -- `\w`'s
# complement also matches Unicode combining marks (category Mn/Mc), and a
# combining vowel sign can legitimately be the last character of an Indic
# script word (e.g. Devanagari matras); using `\w` here would silently trim
# it off the edge. See normalize.py's tokenizer docstring for the same issue.
_EDGE_NOISE_RE = re.compile(
    r"^[\s" + re.escape(string.punctuation) + r"]+|[\s" + re.escape(string.punctuation) + r"]+$"
)


def _collapse_whitespace(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s).strip()


def _clean_component(part: str) -> str:
    stripped = _EDGE_NOISE_RE.sub("", part.strip())
    return _collapse_whitespace(stripped)


@dataclass(frozen=True)
class AddressComponents:
    raw: str
    present: bool
    normalized: str
    components: Tuple[str, ...]
    head: Optional[str]
    middle: Tuple[str, ...]
    tail: Optional[str]


def parse_address(raw: Optional[object]) -> AddressComponents:
    """Decompose one address string into head/middle/tail components.

    Safe on None, NaN, "", and whitespace-only input (returns
    `present=False` with empty components rather than raising or
    fabricating an address).
    """
    raw_str = to_clean_str(raw)
    if raw_str == "":
        return AddressComponents(
            raw="", present=False, normalized="", components=(), head=None, middle=(), tail=None
        )

    nfkc_lower = unicodedata.normalize("NFKC", raw_str).lower()
    normalized = _collapse_whitespace(nfkc_lower)

    components = tuple(
        cleaned for cleaned in (_clean_component(p) for p in normalized.split(",")) if cleaned
    )

    if not components:
        # e.g. raw was just commas/punctuation -- structurally empty even
        # though the raw string itself wasn't blank.
        return AddressComponents(
            raw=raw_str, present=False, normalized=normalized, components=(), head=None,
            middle=(), tail=None,
        )

    if len(components) == 1:
        head = tail = components[0]
        middle: Tuple[str, ...] = ()
    elif len(components) == 2:
        head, tail = components
        middle = ()
    else:
        head = components[0]
        tail = components[-1]
        middle = components[1:-1]

    return AddressComponents(
        raw=raw_str, present=True, normalized=normalized, components=components,
        head=head, middle=middle, tail=tail,
    )
