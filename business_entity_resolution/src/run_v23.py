"""v2.3 in one command: cross-encoder round 3 for France -> gated stacker ->
validated submission files.

    python run_v23.py

  1  export round-3 data: India/US replay + France agreement pairs (GBDT + round-2 CE)
     + synthetic French negatives/positives + French S1-S1 negatives (dev_val asserted absent)
  2  continue training the CE from round 2 -> models/cross_encoder_r3 (round 2 kept)
  3  score dev_val + test with it -> val_scores_r3 / test_scores_r3
  4  stacker CV: round-2 + round-3 CE, 5-seed bagging, coherence features
  5  stacker CV: round-3 CE alone, same settings
  6  GATE: floor = round-2 CE + bagging + coherence (CV 0.98903, no round 3 needed).
     A round-3 variant is used if its CV is within 0.0003 of the floor overall AND
     per country (France is the target, only the portal can measure it); otherwise the floor.
  7  write output/ with the winner, then run the official validator

If any round-3 step fails, the floor is built, so the run always ends with
validated submission files. Overall progress bar before every step; each
step shows its own bars.
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
R3_MODEL = REPO / "data" / "processed" / "models" / "cross_encoder_r3"
TOLERANCE = 0.0003
FLOOR = ("_r2", "floor: round-2 CE + bag5 + coherence")


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
        print(f"\n{'=' * 100}\n v2.3 [{bar}] {frac * 100:3.0f}%  step {self.i + 1}/{len(self.minutes)}: {label}\n"
              f" elapsed {fmt(time.time() - self.start)}, rough time left ~{self.total - self.done} min\n{'=' * 100}", flush=True)
        t0 = time.time()
        ok = subprocess.run([sys.executable, "-u", *cmd], cwd=cwd).returncode == 0
        print(f"\n step {self.i + 1}/{len(self.minutes)} {'done' if ok else 'FAILED'} in {fmt(time.time() - t0)}", flush=True)
        self.skip_to(self.i + 1)
        return ok

    def skip_to(self, i: int) -> None:
        self.i, self.done = i, sum(self.minutes[:i])


def cv_result(tags: str) -> dict:
    return json.loads((CE_DIR / f"stack2_cv{tags.replace(',', '')}_bag5_coh.json").read_text())


def main() -> None:
    r = Runner([12, 55, 50, 6, 6, 18, 6])
    stacker = ["stack2.py", "cv", "--bag", "5", "--coherence", "--tag"]
    if not (CE_DIR / "stack2_cv_r2_bag5_coh.json").exists():  # the floor's CV (normally computed already)
        subprocess.run([sys.executable, "-u", *stacker, "_r2"], cwd=SRC)
    floor = cv_result("_r2")

    r3_ok = (
        r.run("export round-3 CE data (replay + France agreement + synthetic + S1-S1 negatives)", ["cross_encoder.py", "export", "--round3", "--entities", "150000"])
        and r.run("train CE round 3 from round 2 (GPU)", ["cross_encoder.py", "train", "--base-model", str(R2_MODEL), "--train-file", "train_r3.parquet",
                                                         "--out-model", str(R3_MODEL), "--lr", "1e-5"])
        and r.run("score dev_val + test with CE round 3 (GPU)", ["cross_encoder.py", "score", "--model", str(R3_MODEL), "--tag", "_r3"])
    )
    candidates = {}
    if r3_ok:
        for tags, label in (("_r2,_r3", "round-2 + round-3 CE"), ("_r3", "round-3 CE alone")):
            if r.run(f"stacker CV: {label} + bag5 + coherence", [*stacker, tags]):
                candidates[tags] = (cv_result(tags), label)
    else:
        print("\n!! round 3 failed -- building the floor (round-2 CE + bag5 + coherence)", flush=True)
    r.skip_to(5)

    def within(res: dict) -> bool:
        return all(res[k] >= floor[k] - TOLERANCE for k in ("ALL", "India", "US"))

    ok_r3 = {t: v for t, v in candidates.items() if within(v[0])}
    tags, label = (max(ok_r3, key=lambda t: ok_r3[t][0]["ALL"]), None) if ok_r3 else (FLOOR[0], FLOOR[1])
    label = label or ok_r3[tags][1]
    chosen = ok_r3[tags][0] if ok_r3 else floor
    print(f"\n GATE (tolerance {TOLERANCE}): floor CV ALL {floor['ALL']:.5f} (India {floor['India']:.5f}, US {floor['US']:.5f})", flush=True)
    for t, (res, lab) in candidates.items():
        print(f"   {lab:28s} CV ALL {res['ALL']:.5f} (India {res['India']:.5f}, US {res['US']:.5f})  -> {'eligible' if t in ok_r3 else 'rejected'}", flush=True)
    print(f"   USING: {label}", flush=True)

    if not r.run(f"write submission ({label})", ["stack2.py", "apply", "--bag", "5", "--coherence", "--tag", tags]):
        sys.exit("\n!! writing the submission failed -- nothing to upload; send me the error above")
    valid = r.run("official validator", ["utils/validate_submission.py", "--matching", "output/matching_results.tsv",
                                        "--candidate", "output/candidate_pairs.tsv", "--test-dir", "data/raw/test", "--check-ids"], cwd=REPO)
    print(f"\n{'=' * 100}\n v2.3 [{'#' * 30}] 100%  finished in {fmt(time.time() - r.start)}\n"
          f" built: {label}  |  dev_val CV ALL {chosen['ALL']:.5f}, India {chosen['India']:.5f}, US {chosen['US']:.5f}  "
          f"(v2.2 was 0.98884)\n validator: {'PASS -> upload output/matching_results.tsv' if valid else 'FAILED -> do NOT upload; send me the output'}\n"
          f"{'=' * 100}", flush=True)


if __name__ == "__main__":
    main()
