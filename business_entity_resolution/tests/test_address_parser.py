"""Unit tests for src/address_parser.py. Run: python business_entity_resolution/tests/test_address_parser.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from address_parser import parse_address  # noqa: E402


def test_null_handling():
    for blank in (None, float("nan"), "", "   "):
        a = parse_address(blank)
        assert a.present is False
        assert a.components == ()
        assert a.head is None
        assert a.middle == ()
        assert a.tail is None


def test_structurally_empty_despite_nonblank_raw():
    # Only commas/punctuation -- not blank as a raw string, but no usable
    # component once split.
    a = parse_address(",, ,")
    assert a.present is False
    assert a.components == ()


def test_single_component():
    a = parse_address("Just One Field")
    assert a.present is True
    assert a.components == ("just one field",)
    assert a.head == "just one field"
    assert a.tail == "just one field"
    assert a.middle == ()


def test_two_components():
    a = parse_address("Main Street, Springfield")
    assert a.components == ("main street", "springfield")
    assert a.head == "main street"
    assert a.tail == "springfield"
    assert a.middle == ()


def test_us_style_three_component_address():
    a = parse_address("1795 Westchester Drive, High Point, NC")
    assert a.components == ("1795 westchester drive", "high point", "nc")
    assert a.head == "1795 westchester drive"
    assert a.tail == "nc"
    assert a.middle == ("high point",)


def test_reordered_us_address():
    # Real Source-1 example: state/city lead, street trails.
    a = parse_address("OH, Columbus, 5559 Orville Avenue")
    assert a.head == "oh"
    assert a.tail == "5559 orville avenue"
    assert a.middle == ("columbus",)


def test_indian_multi_component_address():
    # Real Source-2 example: 6 comma-separated components with landmarks.
    raw = (
        "HN 247 A/109, MINI MARKET CO-OP. HSG. SOCIETY LTD., FIRST FLOOR, "
        "B. P. ROAD, BHAYANDER (E), THANE, Maharashtra"
    )
    a = parse_address(raw)
    assert a.present is True
    assert len(a.components) == 7
    assert a.head == "hn 247 a/109"
    assert a.tail == "maharashtra"
    assert len(a.middle) == 5


def test_eight_plus_component_address():
    raw = ", ".join(f"part{i}" for i in range(10))
    a = parse_address(raw)
    assert len(a.components) == 10
    assert a.head == "part0"
    assert a.tail == "part9"
    assert len(a.middle) == 8


def test_repeated_commas_and_whitespace_noise():
    a = parse_address("  a St ,, , b City ,   c State   ")
    assert a.components == ("a st", "b city", "c state")
    assert a.head == "a st"
    assert a.tail == "c state"


def test_country_independence_no_branching():
    # Same parser, same code path, regardless of which country the record
    # is for -- including one never seen in the calling code (France).
    us = parse_address("2621 Cotten Road, Tyler, TX")
    india = parse_address("H.no 8, Sector-e, Lucknow")
    france = parse_address("12 Rue de Paris, Lyon, 69000")
    for a in (us, india, france):
        assert a.present is True
        assert a.head is not None
        assert a.tail is not None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
