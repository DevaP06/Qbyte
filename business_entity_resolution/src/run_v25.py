"""v2.5 in one command: multilingual-e5-base cross-encoder -> gated stacker ->
validated submission files. Run after v2.3 (it builds on v2.3's chosen stacker).

    python run_v25.py                      # training budget 140 min (default)
    python run_v25.py --budget-min 100     # if the deadline is tighter

  1  benchmark 200 e5-base steps -> training pairs that fit the budget (go/no-go)
  2  export that many pairs from CE rounds 1-3 (French agreement refreshed with the round-3 CE)
  3  train e5-base (GPU)
  4  score only pairs GBDT and the round-3 CE do not both call confident matches (GPU)
  5  stacker CV: v2.3's configuration + the e5-base score
  6  GATE: e5-base is used only if its CV is within 0.0003 of v2.3's CV overall AND per
     country; otherwise v2.3 is rebuilt unchanged
  7  write output/ with the winner, then run the official validator

Any failure before step 7 falls back to v2.3, so the run always ends with
validated submission files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
CE_DIR = REPO / "data" / "processed" / "cross_encoder"
BASE_OUT = REPO / "data" / "processed" / "models" / "cross_encoder_base"
TOLERANCE = 0.0003


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Runner:
    def __init__(self, minutes: list):
        self.minutes, self.total, self.done, self.i, self.start = minutes, sum(minutes), 0, 0, time.time()

    def run(self, label: str, cmd: list, cwd: Path = SRC) -> bool:
        frac = self.done / self.total
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        print(f"\n{'=' * 100}\n v2.5 [{bar}] {frac * 100:3.0f}%  step {self.i + 1}/{len(self.minutes)}: {label}\n"
              f" elapsed {fmt(time.time() - self.start)}, rough time left ~{self.total - self.done} min\n{'=' * 100}", flush=True)
        t0 = time.time()
        ok = subprocess.run([sys.executable, "-u", *cmd], cwd=cwd).returncode == 0
        print(f"\n step {self.i + 1}/{len(self.minutes)} {'done' if ok else 'FAILED'} in {fmt(time.time() - t0)}", flush=True)
        self.skip_to(self.i + 1)
        return ok

    def skip_to(self, i: int) -> None:
        self.i, self.done = i, sum(self.minutes[:i])


def v23_config() -> tuple:
    """(tags, CV result) of the stacker v2.3 used: its most recently written config."""
    cfg = max(CE_DIR.glob("stack2_config*_bag5_coh.json"), key=lambda p: p.stat().st_mtime)
    tags = ",".join(json.loads(cfg.read_text())["ce_tags"])
    cv = json.loads((CE_DIR / f"stack2_cv{tags.replace(',', '')}_bag5_coh.json").read_text())
    return tags, cv


def main() -> None:
    parser = argparse.ArgumentParser(description="v2.5: e5-base cross-encoder, gated.")
    parser.add_argument("--budget-min", type=float, default=140, help="Training time budget (minutes).")
    args = parser.parse_args()

    tags, base_cv = v23_config()
    print(f" v2.3 stacker: CE {tags} + bag5 + coherence, CV ALL {base_cv['ALL']:.5f} "
          f"(India {base_cv['India']:.5f}, US {base_cv['US']:.5f})", flush=True)
    r = Runner([5, 10, int(args.budget_min), 50, 8, 20, 6])
    stack = ["stack2.py", "--bag", "5", "--coherence", "--tag", tags]
    ok = (
        r.run(f"benchmark e5-base (budget {args.budget_min:.0f} min)", ["ce_base.py", "benchmark", "--budget-min", str(args.budget_min)])
        and r.run("export e5-base training pairs", ["ce_base.py", "export"])
        and r.run("train e5-base cross-encoder (GPU)", ["cross_encoder.py", "train", "--base-model", "intfloat/multilingual-e5-base",
                                                       "--train-file", "train_base.parquet", "--out-model", str(BASE_OUT), "--lr", "2e-5"])
        and r.run("score uncertain pairs with e5-base (GPU)", ["ce_base.py", "score"])
        and r.run("stacker CV: v2.3 + e5-base score", ["stack2.py", "cv", *stack[1:], "--extra-tag", "_base"])
    )
    use_base = False
    if ok:
        cv = json.loads((CE_DIR / f"stack2_cv{tags.replace(',', '')}_bag5_coh_base.json").read_text())
        use_base = all(cv[k] >= base_cv[k] - TOLERANCE for k in ("ALL", "India", "US"))
        print(f"\n GATE (tolerance {TOLERANCE}): v2.3 CV ALL {base_cv['ALL']:.5f} | + e5-base CV ALL {cv['ALL']:.5f} "
              f"(India {cv['India']:.5f}, US {cv['US']:.5f}) -> {'USING e5-base (v2.5)' if use_base else 'rejected, rebuilding v2.3'}", flush=True)
    else:
        print("\n!! an e5-base step failed -- rebuilding v2.3", flush=True)
    r.skip_to(5)

    extra = ["--extra-tag", "_base"] if use_base else []
    if not r.run(f"write submission ({'v2.5 with e5-base' if use_base else 'v2.3'})", ["stack2.py", "apply", *stack[1:], *extra]):
        sys.exit("\n!! writing the submission failed -- nothing to upload; send me the error above")
    valid = r.run("official validator", ["utils/validate_submission.py", "--matching", "output/matching_results.tsv",
                                        "--candidate", "output/candidate_pairs.tsv", "--test-dir", "data/raw/test", "--check-ids"], cwd=REPO)
    chosen = cv if use_base else base_cv
    print(f"\n{'=' * 100}\n v2.5 [{'#' * 30}] 100%  finished in {fmt(time.time() - r.start)}\n"
          f" built: {'v2.5 (+ e5-base)' if use_base else 'v2.3 (e5-base rejected)'}  |  dev_val CV ALL {chosen['ALL']:.5f}, "
          f"India {chosen['India']:.5f}, US {chosen['US']:.5f}\n"
          f" validator: {'PASS -> upload output/matching_results.tsv' if valid else 'FAILED -> do NOT upload; send me the output'}\n{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
