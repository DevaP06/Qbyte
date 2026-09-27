"""e5-base cross-encoder on a second GPU machine, in one command.

    python run_e5_remote.py --budget-min 110      # train (fits the budget) + score if the round-3 files are there
    python run_e5_remote.py --score-only          # after copying val_scores_r3 / test_scores_r3 over

Needs (same paths as the main machine): data/raw/train/train_ground_truth.tsv,
data/processed/{train,test}/ (cleaned records), and in data/processed/cross_encoder/:
train_base.parquet + eval.parquet (training), val_scores_r3.parquet + test_scores_r3.parquet (scoring).

  1  benchmark 200 e5-base steps on this GPU -> pairs that fit --budget-min
  2  trim train_base.parquet to that many pairs
  3  train e5-base (GPU)
  4  score the uncertain dev_val + test pairs (GPU) -- skipped with a message if the
     round-3 score files have not been copied yet (rerun with --score-only)
Copy back: data/processed/cross_encoder/val_scores_base.parquet and test_scores_base.parquet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
CE_DIR = REPO / "data" / "processed" / "cross_encoder"
BASE_OUT = REPO / "data" / "processed" / "models" / "cross_encoder_base"


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def step(i: int, n: int, label: str, cmd: list, start: float, left_min: int) -> bool:
    frac = (i - 1) / n
    bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
    print(f"\n{'=' * 100}\n e5-base [{bar}] {frac * 100:3.0f}%  step {i}/{n}: {label}\n"
          f" elapsed {fmt(time.time() - start)}, rough time left ~{left_min} min\n{'=' * 100}", flush=True)
    t0 = time.time()
    ok = subprocess.run([sys.executable, "-u", *cmd], cwd=SRC).returncode == 0
    print(f"\n step {i}/{n} {'done' if ok else 'FAILED'} in {fmt(time.time() - t0)}", flush=True)
    return ok


def score(start: float) -> None:
    missing = [f for f in ("val_scores_r3.parquet", "test_scores_r3.parquet") if not (CE_DIR / f).exists()]
    if missing:
        print(f"\n!! scoring needs {missing} from the main machine (written when its v2.3 run finishes).\n"
              f"   Copy them into {CE_DIR}, then run:  python run_e5_remote.py --score-only", flush=True)
        return
    if step(4, 4, "score uncertain dev_val + test pairs with e5-base (GPU)", ["ce_base.py", "score"], start, 30):
        print(f"\n{'=' * 100}\n e5-base [{'#' * 30}] 100%  done in {fmt(time.time() - start)}\n"
              f" COPY BACK to the main machine (same folder):\n   {CE_DIR / 'val_scores_base.parquet'}\n   {CE_DIR / 'test_scores_base.parquet'}\n{'=' * 100}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="e5-base cross-encoder on a second GPU machine.")
    parser.add_argument("--budget-min", type=float, default=110, help="Training time budget in minutes.")
    parser.add_argument("--score-only", action="store_true", help="Skip training; score with the trained model.")
    args = parser.parse_args()
    start = time.time()
    if args.score_only:
        score(start)
        return

    if not step(1, 4, f"benchmark e5-base on this GPU (budget {args.budget_min:.0f} min)",
                ["ce_base.py", "benchmark", "--budget-min", str(args.budget_min)], start, int(args.budget_min) + 40):
        sys.exit("benchmark failed -- send me the error above")
    pairs = json.loads((CE_DIR / "ce_base_budget.json").read_text())["pairs"]
    full = pd.read_parquet(CE_DIR / "train_base.parquet")
    fit = full.head(min(pairs, len(full)))  # rows are already shuffled
    fit.to_parquet(CE_DIR / "train_base_fit.parquet", index=False)
    print(f"\n step 2/4: training on {len(fit):,} of {len(full):,} exported pairs (fits {args.budget_min:.0f} min)", flush=True)
    if not step(3, 4, "train e5-base cross-encoder (GPU)",
                ["cross_encoder.py", "train", "--base-model", "intfloat/multilingual-e5-base", "--train-file", "train_base_fit.parquet",
                 "--out-model", str(BASE_OUT), "--lr", "2e-5"], start, int(args.budget_min) + 30):
        sys.exit("training failed -- send me the error above")
    score(start)


if __name__ == "__main__":
    main()
