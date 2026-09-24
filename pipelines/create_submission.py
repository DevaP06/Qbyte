"""Packages the competition submission zip.

Builds <team_name>_submission.zip with the required layout:

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md

Usage:
    python pipelines/create_submission.py <team_name>
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_submission(team_name: str) -> Path:
    output_dir = ROOT / "output"
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        if not (output_dir / name).exists():
            raise FileNotFoundError(f"missing {output_dir / name}; run the pipeline first")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / f"{team_name}_submission"
        shutil.copytree(output_dir, staging / "output", ignore=shutil.ignore_patterns(".gitkeep"))
        shutil.copytree(
            ROOT / "business_entity_resolution",
            staging / "code" / "business_entity_resolution",
            ignore=shutil.ignore_patterns(".gitkeep", "__pycache__"),
        )
        shutil.copy(ROOT / "Documentation_template.md", staging / "Documentation_template.md")

        zip_path = ROOT / f"{team_name}_submission"
        archive = shutil.make_archive(str(zip_path), "zip", staging)
        return Path(archive)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python pipelines/create_submission.py <team_name>")
    result = build_submission(sys.argv[1])
    print(f"wrote {result}")
