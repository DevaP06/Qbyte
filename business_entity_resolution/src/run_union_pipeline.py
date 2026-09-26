"""Run the embedding-union pipeline end to end, with an overall progress bar.

    python run_union_pipeline.py                                  # dev model "dev_union", k=20
    python run_union_pipeline.py --force-features --run dev_v11   # cycle 1: recompute features

Steps run in order and stop at the first failure. Each step's own progress
bars show live; this script adds a whole-pipeline bar before every step.
Embedding and candidate steps skip work already cached; features are
recomputed only with --force-features (needed whenever features.py changes);
the competition pass always re-runs (it replaces its own columns).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent


def steps(k: int, run: str, force_features: bool) -> list:
    union = f"union_retriever_k{k}"
    force = ["--force"] if force_features else []
    # (label, script + args, rough minutes on a 32-thread / RTX 2000 Ada box -- only for the ETA)
    return [
        ("embed test + kNN (cached after first run)", ["embeddings.py", "--split", "test", "--k", "50"], 2),
        ("union candidates: train (cached)", ["candidate_generation.py", "--split", "train", "--emb-k", str(k)], 2),
        ("union candidates: test (cached)", ["candidate_generation.py", "--split", "test", "--emb-k", str(k)], 2),
        ("features: train", ["features.py", "--split", "train", "--candidates", union, *force], 25),
        ("features: test", ["features.py", "--split", "test", "--candidates", union, *force], 20),
        ("competition features: train", ["competition_features.py", "--features", f"train_{union}"], 8),
        ("competition features: test", ["competition_features.py", "--features", f"test_{union}"], 7),
        ("train dev model (GPU)", ["train_classifier.py", "--features", f"train_{union}", "--run", run, "--gpu"], 40),
    ]


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed -> union candidates -> features -> competition -> dev model.")
    parser.add_argument("--k", type=int, default=20, help="Embedding neighbours added per S1 entity.")
    parser.add_argument("--run", default="dev_union", help="Model run name under data/processed/models/.")
    parser.add_argument("--force-features", action="store_true", help="Recompute features even if cached.")
    args = parser.parse_args()

    plan = steps(args.k, args.run, args.force_features)
    total_min = sum(m for *_, m in plan)
    start = time.time()
    done_min = 0
    for i, (label, cmd, minutes) in enumerate(plan, start=1):
        frac = done_min / total_min
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        print(
            f"\n{'=' * 100}\n PIPELINE [{bar}] {frac * 100:3.0f}%  step {i}/{len(plan)}: {label}\n"
            f" elapsed {fmt(time.time() - start)}, rough time left ~{total_min - done_min} min\n{'=' * 100}",
            flush=True,
        )
        t0 = time.time()
        result = subprocess.run([sys.executable, "-u", *cmd], cwd=SRC)
        if result.returncode != 0:
            sys.exit(f"\nstep {i} ({label}) failed with exit code {result.returncode} -- fix it and re-run; "
                     f"finished steps are cached.")
        print(f"\n step {i}/{len(plan)} done in {fmt(time.time() - t0)}", flush=True)
        done_min += minutes

    union = f"union_retriever_k{args.k}"
    print(f"\n{'=' * 100}\n PIPELINE [{'#' * 30}] 100%  all {len(plan)} steps done in {fmt(time.time() - start)}\n"
          f" next: python train_classifier.py --mode final --gpu --features train_{union} "
          f"--dev-run {args.run} --run final_{args.run.removeprefix('dev_')}\n{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
