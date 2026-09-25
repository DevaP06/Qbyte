"""Score a features dataset with a trained model and write the submission files
(plan.md steps 5 + 9).

    python predict.py --run final_gpu            # -> output/candidate_pairs.tsv, output/matching_results.tsv

Reads the model and its locked threshold from data/processed/models/<run>/
(train_classifier.py), scores every pair in data/processed/features/<split>/
part by part (never the whole matrix at once), caches the scores to
data/processed/predictions/<run>/<split>.parquet, then applies
postprocess.select_matches and writes both TSVs through export.py.

candidate_pairs.tsv is written from exactly the pairs scored here -- the
features dataset IS the final candidate set the model runs over, as
PROBLEM_STATEMENT.md requires -- not from an earlier blocking file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

import config
from export import export_submission
from features import FEATURE_COLUMNS
from io_utils import read_cleaned_table
from postprocess import select_matches
from train_classifier import LightGBMModel, Progress, XGBoostModel, _stage


def load_model(run_dir: Path):
    """(model, threshold) from a train_classifier.py run directory."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if metrics["features"] != FEATURE_COLUMNS:
        raise SystemExit(f"{run_dir.name} was trained on a different feature list -- retrain it on current features")
    if metrics.get("backend", "lightgbm") == "xgboost":
        booster = xgb.Booster(model_file=str(run_dir / "model.json"))
        model = XGBoostModel(booster, booster.num_boosted_rounds())
    else:
        model = LightGBMModel(lgb.Booster(model_file=str(run_dir / "model.txt")))
    return model, float(metrics["threshold"])


def score_features(model, features_dir: Path) -> pd.DataFrame:
    parts = sorted(features_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no feature parts in {features_dir} -- run features.py first")
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    bar = Progress(total, "scoring", "pairs")
    out, done = [], 0
    for part in parts:
        table = pq.read_table(part, columns=["source1_entity_id", "candidate_entity_id", "country", *FEATURE_COLUMNS])
        X = np.column_stack([table.column(c).to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        out.append(
            pa.table(
                {
                    "source1_entity_id": table.column("source1_entity_id"),
                    "candidate_entity_id": table.column("candidate_entity_id"),
                    "country": table.column("country"),
                    "score": pa.array(model.predict(X).astype(np.float32)),
                }
            )
        )
        done += table.num_rows
        bar.update(done, f"{len(out)}/{len(parts)} parts")
    bar.close()
    return pa.concat_tables(out).to_pandas()


def summarize(source1: pd.DataFrame, scored: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per-country sanity numbers to compare against dev_val / ground truth
    (train: ~5.6% singletons, ~3.5 true matches per entity)."""
    n_cand = scored.groupby("source1_entity_id").size()
    n_match = matches.groupby("source1_entity_id").size()
    s = source1.set_index("entity_id")
    s["n_candidates"] = n_cand.reindex(s.index).fillna(0)
    s["n_matches"] = n_match.reindex(s.index).fillna(0)
    return s.groupby("country").agg(
        entities=("n_matches", "size"),
        no_candidates=("n_candidates", lambda x: int((x == 0).sum())),
        predicted_singletons=("n_matches", lambda x: (x == 0).mean()),
        mean_matches=("n_matches", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates and write the submission TSVs.")
    parser.add_argument("--run", required=True, help="Model run directory under data/processed/models/.")
    parser.add_argument("--split", default="test", choices=["test"], help="Split whose S1 entities get output rows.")
    parser.add_argument("--features", default=None, help="Features directory name (default: same as --split).")
    parser.add_argument("--out-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="Re-score even if cached predictions exist.")
    args = parser.parse_args()
    features_name = args.features or args.split
    t0 = time.time()

    _stage(f"loading model {args.run}")
    model, threshold = load_model(config.MODELS_DIR / args.run)
    print(f"  {type(model).__name__}, {model.n_rounds} rounds, threshold {threshold:.4f}", flush=True)

    cache = config.PREDICTIONS_DIR / args.run / f"{features_name}.parquet"
    # A retrain under the same run name (e.g. dev_gpu re-run) must invalidate
    # cached scores: trust the cache only if it is newer than the model and
    # the features it was computed from.
    run_dir = config.MODELS_DIR / args.run
    inputs = [p for p in run_dir.glob("model.*")] + sorted((config.FEATURES_DIR / features_name).glob("part-*.parquet"))
    fresh = cache.exists() and all(cache.stat().st_mtime > p.stat().st_mtime for p in inputs)
    if fresh and not args.force:
        _stage(f"using cached scores {cache}")
        scored = pd.read_parquet(cache)
    else:
        _stage(f"scoring features/{features_name}")
        scored = score_features(model, config.FEATURES_DIR / features_name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(cache, index=False)

    _stage("post-processing (threshold + one S1 per S2/S3 id)")
    above = int((scored["score"] >= threshold).sum())
    matches = select_matches(scored, threshold)
    print(f"  {above:,} pairs above threshold -> {len(matches):,} after conflict resolution", flush=True)

    _stage(f"writing submission files to {args.out_dir}")
    source1 = read_cleaned_table(args.split, "source1", columns=["entity_id", "country"]).to_pandas()
    export_submission(Path(args.out_dir), source1["entity_id"].to_numpy(), scored, matches)

    pd.set_option("display.width", 200)
    print(summarize(source1, scored, matches).to_string(float_format=lambda x: f"{x:.4f}"))
    _stage(f"done in {time.time() - t0:.0f}s -- now run utils/validate_submission.py")


if __name__ == "__main__":
    main()
