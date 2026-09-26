"""Run the embedding-union pipeline end to end, with an overall progress bar.

    python run_union_pipeline.py            # --k 20 --run dev_union
    python run_union_pipeline.py --k 30

Steps run in order and stop at the first failure. Each step's own progress
bars show live; this script adds a whole-pipeline bar before every step.
Steps that cache their output (embeddings, candidate_generation, features)
skip work already done, so re-running after a failure resumes cheaply.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent


def steps(k: int, run: str) -> list:
    union = f"union_retriever_k{k}"
    # (label, script + args, rough minutes on a 32-thread / RTX 2000 Ada box -- only for the ETA)
    return [
        ("embed test + kNN", ["embeddings.py", "--split", "test", "--k", "50"], 10),
        ("union candidates: train (embeds train first)", ["candidate_generation.py", "--split", "train", "--emb-k", str(k)], 45),
        ("union candidates: test", ["candidate_generation.py", "--split", "test", "--emb-k", str(k)], 10),
        ("features: train", ["features.py", "--split", "train", "--candidates", union], 20),
        ("features: test", ["features.py", "--split", "test", "--candidates", union], 15),
        ("train dev model (GPU)", ["train_classifier.py", "--features", f"train_{union}", "--run", run, "--gpu"], 45),
    ]


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed -> union candidates -> features -> dev model, in one go.")
    parser.add_argument("--k", type=int, default=20, help="Embedding neighbours added per S1 entity.")
    parser.add_argument("--run", default="dev_union", help="Model run name under data/processed/models/.")
    args = parser.parse_args()

    plan = steps(args.k, args.run)
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

    print(f"\n{'=' * 100}\n PIPELINE [{'#' * 30}] 100%  all {len(plan)} steps done in {fmt(time.time() - start)}\n"
          f" next: python train_classifier.py --mode final --gpu --features train_union_retriever_k{args.k} "
          f"--dev-run {args.run} --run final_union\n{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
