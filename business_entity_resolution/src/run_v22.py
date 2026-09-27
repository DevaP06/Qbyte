"""v2.2 in one command: cross-encoder round 2 (+ France agreement pairs) -> gated
level-2 stacker -> validated submission files.

    python run_v22.py

  1  export round-2 CE data: fit entities round 1 never saw + France agreement pairs
  2  continue training the CE from round 1 -> models/cross_encoder_r2 (round 1 kept)
  3  score dev_val + test with it -> val_scores_r2 / test_scores_r2
  4  stacker 2-fold CV with round-1 CE scores   (= v2.1)
  5  stacker 2-fold CV with round-2 CE scores
  6  GATE: round 2 is used only if its CV macro F0.5 beats round 1; otherwise v2.1 is built
  7  write output/ with the winner, then run the official validator

If any round-2 step fails, the run continues with round 1 (v2.1), so it always
ends with validated submission files. Overall progress bar before every step;
each step shows its own bars.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
CE_DIR = REPO / "data" / "processed" / "cross_encoder"
R2_MODEL = REPO / "data" / "processed" / "models" / "cross_encoder_r2"
R1_MODEL = REPO / "data" / "processed" / "models" / "cross_encoder"


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Runner:
    def __init__(self, plan_minutes: list):
        self.total, self.done, self.start, self.i, self.n = sum(plan_minutes), 0, time.time(), 0, len(plan_minutes)
        self.minutes = plan_minutes

    def run(self, label: str, cmd: list, cwd: Path = SRC) -> bool:
        frac = self.done / self.total
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        print(f"\n{'=' * 100}\n v2.2 [{bar}] {frac * 100:3.0f}%  step {self.i + 1}/{self.n}: {label}\n"
              f" elapsed {fmt(time.time() - self.start)}, rough time left ~{self.total - self.done} min\n{'=' * 100}", flush=True)
        t0 = time.time()
        ok = subprocess.run([sys.executable, "-u", *cmd], cwd=cwd).returncode == 0
        print(f"\n step {self.i + 1}/{self.n} {'done' if ok else 'FAILED'} in {fmt(time.time() - t0)}", flush=True)
        self.done += self.minutes[self.i]
        self.i += 1
        return ok


def main() -> None:
    r = Runner([10, 70, 45, 6, 6, 15, 6])
    r2_ok = (
        r.run("export round-2 CE data (new fit entities + France agreement pairs)", ["cross_encoder.py", "export", "--round2", "--entities", "250000"])
        and r.run("train CE round 2 from round 1 (GPU)", ["cross_encoder.py", "train", "--base-model", str(R1_MODEL),
                                                          "--train-file", "train_r2.parquet", "--out-model", str(R2_MODEL), "--lr", "1e-5"])
        and r.run("score dev_val + test with CE round 2 (GPU)", ["cross_encoder.py", "score", "--model", str(R2_MODEL), "--tag", "_r2"])
    )
    if not r2_ok:  # keep the step counter/ETA consistent when round 2 is abandoned early
        print("\n!! round 2 failed -- continuing with round 1 (v2.1)", flush=True)
        r.i, r.done = 3, sum(r.minutes[:3])

    r.run("stacker CV with round-1 CE scores (v2.1)", ["stack2.py", "cv"])
    cv_r1 = json.loads((CE_DIR / "stack2_cv.json").read_text())
    cv_r2 = None
    if r2_ok and r.run("stacker CV with round-2 CE scores", ["stack2.py", "cv", "--tag", "_r2"]):
        cv_r2 = json.loads((CE_DIR / "stack2_cv_r2.json").read_text())
    else:
        r.i, r.done = 5, sum(r.minutes[:5])

    use_r2 = cv_r2 is not None and cv_r2["ALL"] > cv_r1["ALL"]
    tag = "_r2" if use_r2 else ""
    print(f"\n GATE: dev_val CV macro F0.5  round 1 = {cv_r1['ALL']:.4f}   round 2 = "
          f"{cv_r2['ALL'] if cv_r2 else float('nan'):.4f}  ->  using {'ROUND 2 (v2.2)' if use_r2 else 'ROUND 1 (v2.1)'}", flush=True)

    if not r.run(f"write submission with the {'round-2' if use_r2 else 'round-1'} stacker", ["stack2.py", "apply", "--tag", tag]):
        sys.exit("\n!! writing the submission failed -- nothing to upload; see the error above")
    valid = r.run("official validator", ["utils/validate_submission.py", "--matching", "output/matching_results.tsv",
                                        "--candidate", "output/candidate_pairs.tsv", "--test-dir", "data/raw/test", "--check-ids"], cwd=REPO)
    chosen = cv_r2 if use_r2 else cv_r1
    print(f"\n{'=' * 100}\n v2.2 [{'#' * 30}] 100%  finished in {fmt(time.time() - r.start)}\n"
          f" version built: {'v2.2 (CE round 2)' if use_r2 else 'v2.1 (CE round 1)'}  |  dev_val CV: ALL {chosen['ALL']:.4f}, "
          f"India {chosen['India']:.4f}, US {chosen['US']:.4f}\n"
          f" validator: {'PASS -> upload output/matching_results.tsv' if valid else 'FAILED -> do NOT upload; send me the output'}\n{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
