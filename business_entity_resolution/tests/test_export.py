"""Unit tests for src/postprocess.py and src/export.py, including a round trip
through the official utils/validate_submission.py.

Run: python business_entity_resolution/tests/test_export.py
"""

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "utils"))

from export import CANDIDATE_FILE, MATCHING_FILE, export_submission  # noqa: E402
from postprocess import select_matches  # noqa: E402
from validate_submission import validate  # noqa: E402

SOURCE1 = np.array(["S1-1", "S1-2", "S1-3", "S1-4"], dtype=object)  # S1-4: no candidates at all

SCORED = pd.DataFrame(
    [
        ("S1-1", "S2-10", 0.95),
        ("S1-1", "S3-11", 0.90),
        ("S1-1", "S2-12", 0.10),
        ("S1-2", "S2-10", 0.80),  # S2-10 also wanted by S1-1 at 0.95 -> S1-1 keeps it
        ("S1-2", "S3-20", 0.85),
        ("S1-3", "S2-30", 0.20),  # below threshold -> S1-3 predicted singleton
    ],
    columns=["source1_entity_id", "candidate_entity_id", "score"],
)


def test_select_matches_threshold_and_conflict_resolution():
    m = select_matches(SCORED, threshold=0.5)
    got = set(zip(m["source1_entity_id"], m["candidate_entity_id"]))
    assert got == {("S1-1", "S2-10"), ("S1-1", "S3-11"), ("S1-2", "S3-20")}, got


def test_select_matches_tie_is_deterministic():
    tie = pd.DataFrame(
        [("S1-9", "S2-1", 0.7), ("S1-8", "S2-1", 0.7)], columns=["source1_entity_id", "candidate_entity_id", "score"]
    )
    m = select_matches(tie, threshold=0.5)
    assert list(m["source1_entity_id"]) == ["S1-8"]  # lower S1 id wins a tie


def _export(tmp: Path) -> Path:
    out = tmp / "output"
    export_submission(out, SOURCE1, SCORED, select_matches(SCORED, threshold=0.5))
    return out


def test_file_contents_and_format():
    with tempfile.TemporaryDirectory() as tmp:
        out = _export(Path(tmp))
        raw = (out / MATCHING_FILE).read_bytes()
        assert b"\r" not in raw, "Windows line endings would corrupt the last id of every row"
        lines = raw.decode("utf-8").split("\n")
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1:5] == ["S1-1\tS2-10,S3-11", "S1-2\tS3-20", "S1-3\t", "S1-4\t"]  # score order, empties kept
        cand = (out / CANDIDATE_FILE).read_text(encoding="utf-8").split("\n")
        assert cand[0] == "source1_entity_id\tcandidate_entity_ids"
        assert cand[1] == "S1-1\tS2-10,S3-11,S2-12"
        assert cand[4] == "S1-4\t"


def test_official_validator_passes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_dir = tmp / "test"
        test_dir.mkdir()
        (test_dir / "test_source1.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            + "".join(f"{s}\tname\taddr\tUS\n" for s in SOURCE1),
            encoding="utf-8",
        )
        out = _export(tmp)
        with redirect_stdout(io.StringIO()):
            errors, _ = validate(str(out / MATCHING_FILE), str(out / CANDIDATE_FILE), str(test_dir))
        assert errors == [], errors


def test_match_outside_candidates_is_rejected():
    bogus = pd.DataFrame([("S1-1", "S2-999", 0.9)], columns=["source1_entity_id", "candidate_entity_id", "score"])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            export_submission(Path(tmp), SOURCE1, SCORED, bogus)
        except AssertionError:
            return
    raise AssertionError("export accepted a match that was never a candidate")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__}: ok")
    print("ok")


if __name__ == "__main__":
    main()
