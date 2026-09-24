"""Smoke test for pipelines/create_submission.py. Run: python tests/test_create_submission.py"""

import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipelines"))

import create_submission  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake_root = Path(tmp)
        create_submission.ROOT = fake_root

        (fake_root / "output").mkdir()
        (fake_root / "output" / "matching_results.tsv").write_text("id\tmatch\n")
        (fake_root / "output" / "candidate_pairs.tsv").write_text("id1\tid2\n")

        ber = fake_root / "business_entity_resolution"
        (ber / "src").mkdir(parents=True)
        (ber / "README.md").write_text("# stub\n")
        (ber / "requirements.txt").write_text("\n")

        (fake_root / "Documentation_template.md").write_text("# doc\n")

        zip_path = create_submission.build_submission("testteam")
        names = set(zipfile.ZipFile(zip_path).namelist())

        assert "output/matching_results.tsv" in names
        assert "output/candidate_pairs.tsv" in names
        assert "code/business_entity_resolution/README.md" in names
        assert "code/business_entity_resolution/requirements.txt" in names
        assert "Documentation_template.md" in names

    print("ok")


if __name__ == "__main__":
    main()
