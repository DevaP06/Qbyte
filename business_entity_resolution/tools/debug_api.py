"""FastAPI tool for inspecting every pipeline step for one entity or pair.

    python business_entity_resolution/tools/debug_api.py      # then open http://127.0.0.1:8000/docs

Endpoints (split = train | test):
  /record/{split}/{entity_id}         step 0-1: raw TSV row and its cleaned form
  /entity/{split}/{s1_id}             steps 2-5 for an S1 entity: token-blocking and embedding
                                      candidates, model score + decision per candidate, and
                                      (train) ground truth incl. true matches never retrieved
  /pair/{split}/{s1_id}/{cand_id}     every feature value + model score for one pair
  /errors                             rows of error_analysis.py's file, filterable

Read-only; loads lazily and caches (first call per split is slow: it indexes
which feature part holds each S1 entity). Defaults: model run dev_union for
train (its dev_val entities are honest), final_union for test.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from features import feature_columns_of  # noqa: E402
from labels import ground_truth_pairs, load_ground_truth  # noqa: E402
from predict import load_model  # noqa: E402

UNION = "union_retriever_k20"
DEFAULT_RUN = {"train": "dev_v11", "test": "final_v11"}  # latest models matching the current features
SOURCES = ("source1", "source2", "source3")

app = FastAPI(title="Qbyte entity-resolution debugger")


def _json(df: pd.DataFrame) -> list:
    # pandas' encoder handles numpy scalars, token-list arrays, and NaN -> null
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _check_split(split: str) -> None:
    if split not in ("train", "test"):
        raise HTTPException(404, "split must be train or test")


def _source_of(entity_id: str) -> str:
    return {"S1": "source1", "S2": "source2", "S3": "source3"}[entity_id[:2]]


@lru_cache(maxsize=6)
def _raw(split: str, source: str) -> pd.DataFrame:
    files = config.TRAIN_FILES if split == "train" else config.TEST_FILES
    raw_dir = config.DATA_RAW_TRAIN_DIR if split == "train" else config.DATA_RAW_TEST_DIR
    t = pacsv.read_csv(raw_dir / files[source], parse_options=pacsv.ParseOptions(delimiter="\t"),
                       convert_options=pacsv.ConvertOptions(strings_can_be_null=False))
    return t.to_pandas().set_index("entity_id")


@lru_cache(maxsize=6)
def _clean(split: str, source: str) -> pd.DataFrame:
    return pq.read_table(config.DATA_PROCESSED_DIR / split / f"{source}.parquet").to_pandas().set_index("entity_id")


def _records(split: str, ids) -> pd.DataFrame:
    """Cleaned name/address/country for a list of ids (any source)."""
    out = []
    for source in SOURCES:
        want = [i for i in ids if _source_of(i) == source]
        if want:
            c = _clean(split, source)
            out.append(c.reindex(want)[["country", "name_original_normalized", "address_normalized"]])
    return pd.concat(out) if out else pd.DataFrame()


@lru_cache(maxsize=2)
def _part_index(split: str) -> dict:
    """S1 id -> feature part file (features are written S1-grouped, so one part per entity)."""
    index = {}
    for part in sorted((config.FEATURES_DIR / f"{split}_{UNION}").glob("part-*.parquet")):
        for s1 in pq.read_table(part, columns=["source1_entity_id"]).column(0).unique().to_pylist():
            index[s1] = part
    return index


@lru_cache(maxsize=4)
def _model(run: str):
    return load_model(config.MODELS_DIR / run)


@lru_cache(maxsize=1)
def _truth() -> pd.DataFrame:
    return ground_truth_pairs(load_ground_truth())


def _scored_features(split: str, s1_id: str, run: str) -> pd.DataFrame:
    part = _part_index(split).get(s1_id)
    if part is None:
        return pd.DataFrame()
    feats = pq.read_table(part, filters=[("source1_entity_id", "==", s1_id)]).to_pandas()
    model, threshold, names = _model(run)
    if names != feature_columns_of(part.parent):
        raise HTTPException(409, f"model {run} was trained on different features than {part.parent.name}")
    feats["score"] = model.predict(feats[names].to_numpy(np.float32))
    feats["above_threshold"] = feats["score"] >= threshold
    return feats.sort_values("score", ascending=False)


@app.get("/record/{split}/{entity_id}")
def record(split: str, entity_id: str):
    _check_split(split)
    source = _source_of(entity_id)
    raw, clean = _raw(split, source), _clean(split, source)
    if entity_id not in clean.index:
        raise HTTPException(404, f"{entity_id} not in {split}/{source}")
    return {"raw": _json(raw.loc[[entity_id]])[0], "cleaned": _json(clean.loc[[entity_id]])[0]}


@app.get("/entity/{split}/{s1_id}")
def entity(split: str, s1_id: str, run: Optional[str] = None, top: int = 60):
    _check_split(split)
    run = run or DEFAULT_RUN[split]
    s1 = _records(split, [s1_id])
    if s1.empty or s1.iloc[0].isna().all():
        raise HTTPException(404, f"{s1_id} not in {split}")
    cand_dir = config.CANDIDATES_DIR / split
    token = pq.read_table(cand_dir / "token_blocking.parquet", filters=[("source1_entity_id", "==", s1_id)]).to_pandas()
    emb = pq.read_table(cand_dir / "embedding_retriever.parquet", filters=[("source1_entity_id", "==", s1_id)]).to_pandas()
    scored = _scored_features(split, s1_id, run)
    _, threshold, _ = _model(run)

    true_ids = set(_truth().query("source1_entity_id == @s1_id")["matched_id"]) if split == "train" else set()
    view = scored.head(top)[["candidate_entity_id", "score", "above_threshold", "block_rank", "from_token",
                             "emb_cos", "emb_rank", "name_token_set_ratio", "addr_token_set_ratio"]].copy()
    view["is_true_match"] = view["candidate_entity_id"].isin(true_ids) if split == "train" else None
    texts = _records(split, view["candidate_entity_id"].tolist())
    view["name"] = texts["name_original_normalized"].reindex(view["candidate_entity_id"]).to_numpy()
    view["address"] = texts["address_normalized"].reindex(view["candidate_entity_id"]).to_numpy()

    out = {
        "s1": _json(s1.reset_index())[0],
        "model": {"run": run, "threshold": threshold},
        "counts": {"token_candidates": len(token), "embedding_candidates": len(emb), "scored_union": len(scored),
                   "predicted_matches_before_conflict_resolution": int(scored["above_threshold"].sum()) if len(scored) else 0},
        "candidates": _json(view),
    }
    if split == "train":
        retrieved = set(scored["candidate_entity_id"]) if len(scored) else set()
        missed = sorted(true_ids - retrieved)
        out["ground_truth"] = {"true_matches": sorted(true_ids), "never_retrieved": _json(_records(split, missed).reset_index()) if missed else []}
    return out


@app.get("/pair/{split}/{s1_id}/{cand_id}")
def pair(split: str, s1_id: str, cand_id: str, run: Optional[str] = None):
    _check_split(split)
    run = run or DEFAULT_RUN[split]
    scored = _scored_features(split, s1_id, run)
    row = scored[scored["candidate_entity_id"] == cand_id] if len(scored) else scored
    if row.empty:
        raise HTTPException(404, f"({s1_id}, {cand_id}) is not a candidate pair in {split}")
    texts = _records(split, [s1_id, cand_id])
    return {"s1": _json(texts.loc[[s1_id]].reset_index())[0], "candidate": _json(texts.loc[[cand_id]].reset_index())[0],
            "model": {"run": run, "threshold": _model(run)[1]}, "pair": _json(row)[0]}


@lru_cache(maxsize=2)
def _errors(run: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED_DIR / "errors" / f"{run}_errors.parquet"
    if not path.exists():
        raise HTTPException(404, f"{path} missing -- run: python error_analysis.py --run {run}")
    return pd.read_parquet(path)


@app.get("/errors")
def errors(run: str = "dev_union", error_type: Optional[str] = None, country: Optional[str] = None,
           no_candidate_address: Optional[bool] = None, limit: int = 50, offset: int = 0):
    e = _errors(run)
    if error_type:
        e = e[e["error_type"] == error_type.upper()]
    if country:
        e = e[e["country"] == country]
    if no_candidate_address is not None:
        e = e[~e["candidate_text"].fillna("").str.contains(",") == no_candidate_address]
    return {"total": len(e), "rows": _json(e.iloc[offset : offset + limit])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
