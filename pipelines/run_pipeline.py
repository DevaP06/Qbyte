"""Runs the entity resolution pipeline end to end.

Blocking/matching logic not yet implemented. MLflow tracking is wired up
so every run (once implemented) is reproducible from three coordinates:
the git commit (code), the data_source.yaml version_ids (data), and the
logged MLflow params/metrics/model (run).
"""

import subprocess
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()


def _data_versions() -> dict:
    config = yaml.safe_load((ROOT / "configs" / "data_source.yaml").read_text())
    return {f"data.{name}.version_id": path["version_id"] for name, path in config["paths"].items()}


def main() -> None:
    with mlflow.start_run():
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_params(_data_versions())
        raise NotImplementedError("Pipeline not implemented yet; blocking/matching approach not decided.")


if __name__ == "__main__":
    main()
