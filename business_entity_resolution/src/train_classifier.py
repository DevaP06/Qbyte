"""Pair classifier + threshold tuning + LOCO diagnostic (plan.md step 4).

LightGBM on CPU by default; `--gpu` trains XGBoost on CUDA instead (see the
backend notes in the Training section for why not LightGBM on GPU).

Modes:

  dev    (default) Split train S1 entities into fit / early-stopping /
         dev_val (entity-level, stratified by country), train on fit, tune the
         global macro-F_0.5 threshold on dev_val, report dev_val F_0.5 overall
         and per country. With --loco, also runs the leave-one-country-out
         stress test (plan.md "the France risk").
  final  Retrain on ALL train entities for the dev run's best iteration
         count, carrying its locked threshold forward unchanged (plan.md
         "Final model").

Artifacts go to data/processed/models/<run>/: model.txt (LightGBM) or
model.json (XGBoost), metrics.json, feature_importance.csv,
threshold_curve.csv, and (dev) val_predictions.parquet for error analysis and
post-processing work.

The full feature matrix is loaded once into one preallocated float32 array;
no fold copies it (LightGBM: `Dataset.subset` of one binned dataset;
XGBoost: QuantileDMatrix fed in row batches).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

import config
from evaluate import macro_f05, macro_f05_by_group
from features import FEATURE_COLUMNS
from labels import PairLabeler, entity_truth, ground_truth_pairs, load_ground_truth
from threshold_tuning import ThresholdResult, tune_threshold

ROLE_FIT, ROLE_EARLY_STOP, ROLE_VAL = 0, 1, 2


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _stage(message: str) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Progress:
    """One live progress line with elapsed time, rate, and ETA.

    On a terminal it redraws in place (carriage return, twice a second). When
    stdout is piped or redirected -- e.g. `| Tee-Object run.log` -- a bare
    carriage return never completes a line, so nothing would show until the
    run ended; there it prints one plain line every `log_every` seconds.
    """

    def __init__(self, total: int, label: str, unit: str, eta_is_upper_bound: bool = False, log_every: float = 30.0):
        self.total, self.label, self.unit = total, label, unit
        self.eta_prefix = "ETA<=" if eta_is_upper_bound else "ETA"
        self.tty = sys.stdout.isatty()
        self.min_interval = 0.5 if self.tty else log_every
        self.start = time.time()
        self.last_print = 0.0
        self.last_len = 0
        self.done = 0
        self.suffix = ""

    def update(self, done: int, suffix: str = "") -> None:
        self.done, self.suffix = done, suffix
        now = time.time()
        if now - self.last_print >= self.min_interval:
            self._print(now)

    def close(self, suffix: Optional[str] = None) -> None:
        if suffix is not None:
            self.suffix = suffix
        self._print(time.time())
        if self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _print(self, now: float) -> None:
        self.last_print = now
        frac = min(self.done / self.total, 1.0) if self.total else 1.0
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        line = (
            f"  {self.label} [{bar}] {frac * 100:5.1f}%  {self.done:,}/{self.total:,} {self.unit}  "
            f"{_fmt_duration(elapsed)} elapsed, {self.eta_prefix} {_fmt_duration(eta)}"
            + (f"  | {self.suffix}" if self.suffix else "")
        )
        if self.tty:
            sys.stdout.write("\r" + line.ljust(self.last_len))
            self.last_len = len(line)
        else:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()


class _RoundTracker:
    """Shared by both backends' training callbacks: feeds a Progress bar with
    the round count and the early-stopping metric (best value, its round, and
    rounds since -- training stops at `patience` rounds without improvement)."""

    def __init__(self, label: str, total_rounds: int, early_stopping: bool):
        self.bar = Progress(total_rounds, label, "rounds", eta_is_upper_bound=early_stopping)
        self.watch = "early_stop" if early_stopping else "fit"
        self.best, self.best_round = float("inf"), 0

    def update(self, round_1based: int, metrics: dict) -> None:
        value = metrics.get(self.watch)
        if value is None:
            self.bar.update(round_1based)
            return
        if value < self.best:
            self.best, self.best_round = value, round_1based
        suffix = f"{self.watch} logloss {value:.5f}"
        if self.watch == "early_stop":
            since = round_1based - self.best_round
            suffix += f" (best {self.best:.5f} @ {self.best_round}, {since}/{config.EARLY_STOPPING_ROUNDS} w/o gain)"
        self.bar.update(round_1based, suffix)

    def close(self) -> None:
        self.bar.close()


class _XGBoostProgress(xgb.callback.TrainingCallback):
    def __init__(self, tracker: _RoundTracker):
        self.tracker = tracker
        super().__init__()

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        metrics = {name: log["logloss"][-1] for name, log in evals_log.items() if "logloss" in log}
        self.tracker.update(epoch + 1, metrics)
        return False  # never request a stop; early stopping is xgboost's own callback


def _lightgbm_progress(tracker: _RoundTracker):
    def callback(env: lgb.callback.CallbackEnv) -> None:
        metrics = {name: value for name, metric, value, _ in env.evaluation_result_list if metric == "binary_logloss"}
        tracker.update(env.iteration + 1, metrics)

    return callback


# --------------------------------------------------------------------------
# Entity split
# --------------------------------------------------------------------------


def split_entities(
    countries: np.ndarray,
    val_fraction: float = config.DEV_VAL_FRACTION,
    early_stop_fraction: float = config.EARLY_STOPPING_FRACTION,
    seed: int = config.SPLIT_SEED,
) -> np.ndarray:
    """Per-entity role (fit / early-stop / val), stratified by country.
    Deterministic for a given seed and entity order."""
    rng = np.random.default_rng(seed)
    roles = np.full(len(countries), ROLE_FIT, dtype=np.int8)
    for country in np.unique(countries):
        idx = rng.permutation(np.flatnonzero(countries == country))
        n_val = int(round(len(idx) * val_fraction))
        n_es = int(round(len(idx) * early_stop_fraction))
        roles[idx[:n_val]] = ROLE_VAL
        roles[idx[n_val : n_val + n_es]] = ROLE_EARLY_STOP
    return roles


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class TrainingData:
    X: np.ndarray  # (n_pairs, n_features) float32, columns = FEATURE_COLUMNS
    y: np.ndarray  # (n_pairs,) int8
    entity: np.ndarray  # (n_pairs,) int32 index into `truth`
    candidate_ids: pa.ChunkedArray
    truth: pd.DataFrame  # every train S1 entity: source1_entity_id, country, n_true
    # Per-entity: which entities the run may train/evaluate on. All of them
    # for the full dataset (candidate-less entities must still count as 0 in
    # macro F_0.5); only the sampled ones for a --sample-entities dataset,
    # else every unsampled dev_val entity would score as a total miss.
    scope: np.ndarray


def load_training_data(
    features_dir: Path, truth: pd.DataFrame, labeler: PairLabeler, sampled: bool = False
) -> TrainingData:
    parts = sorted(features_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no feature parts in {features_dir} -- run features.py first")
    n_total = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    X = np.empty((n_total, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty(n_total, dtype=np.int8)
    entity = np.empty(n_total, dtype=np.int32)
    candidate_chunks = []
    entity_index = pd.Index(truth["source1_entity_id"].to_numpy())

    bar = Progress(n_total, "loading", "pairs")
    offset = 0
    for part in parts:
        table = pq.read_table(part, columns=["source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS])
        n = table.num_rows
        s1_ids = table.column("source1_entity_id").to_numpy()
        cand = table.column("candidate_entity_id").combine_chunks()
        for j, col in enumerate(FEATURE_COLUMNS):
            X[offset : offset + n, j] = table.column(col).to_numpy()  # nulls -> NaN
        codes = entity_index.get_indexer(s1_ids)
        assert (codes >= 0).all(), f"{part.name}: S1 id not in ground truth"
        entity[offset : offset + n] = codes
        y[offset : offset + n] = labeler.label(s1_ids, cand.to_numpy(zero_copy_only=False))
        candidate_chunks.append(cand)
        offset += n
        bar.update(offset, f"{len(candidate_chunks)}/{len(parts)} parts")
    bar.close()
    scope = np.bincount(entity, minlength=len(truth)) > 0 if sampled else np.ones(len(truth), dtype=bool)
    return TrainingData(
        X=X, y=y, entity=entity, candidate_ids=pa.chunked_array(candidate_chunks), truth=truth, scope=scope
    )


# --------------------------------------------------------------------------
# Training / scoring
# --------------------------------------------------------------------------


# Two interchangeable backends behind one small interface -- a trainer's
# fit(fit_rows, early_stop_rows, num_rounds) returns a model with
# `n_rounds` (boosting rounds kept, a count), predict(X), save(dir), and
# importance(). Everything else (split, threshold, LOCO, artifacts) is shared.
#
#   LightGBM (CPU, default): the plan's primary model.
#   XGBoost on CUDA (--gpu): the plan's named alternative. Used for GPU
#     training because the PyPI LightGBM wheel is built without GPU support
#     ("GPU/CUDA Tree Learner was not enabled in this build"), whereas the
#     XGBoost wheel ships with CUDA.


class LightGBMModel:
    def __init__(self, booster: lgb.Booster):
        self.booster = booster
        self.n_rounds = booster.best_iteration or booster.current_iteration()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.n_rounds)

    def save(self, out_dir: Path) -> None:
        self.booster.save_model(str(out_dir / "model.txt"), num_iteration=self.n_rounds)

    def importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.booster.feature_name(),
                "gain": self.booster.feature_importance("gain"),
                "split": self.booster.feature_importance("split"),
            }
        ).sort_values("gain", ascending=False)


class LightGBMTrainer:
    name = "lightgbm"
    params = config.LGBM_PARAMS

    def __init__(self, data: TrainingData):
        # Binned once; every fold trains on a .subset() of it, so no fold
        # copies the raw matrix. free_raw_data=False keeps subsetting possible.
        self.full = lgb.Dataset(
            data.X, label=data.y, feature_name=FEATURE_COLUMNS, free_raw_data=False,
            params={"max_bin": config.LGBM_PARAMS["max_bin"], "verbose": -1},
        ).construct()

    def fit(
        self,
        fit_rows: np.ndarray,
        early_stop_rows: Optional[np.ndarray],
        num_rounds: int = config.MAX_ROUNDS,
        label: str = "training",
    ) -> LightGBMModel:
        train_set = self.full.subset(fit_rows.astype(np.int32))
        tracker = _RoundTracker(label, num_rounds, early_stopping=early_stop_rows is not None)
        callbacks = [_lightgbm_progress(tracker)]
        valid_sets, valid_names = [train_set], ["fit"]
        if early_stop_rows is not None:
            valid_sets.append(self.full.subset(early_stop_rows.astype(np.int32)))
            valid_names.append("early_stop")
            callbacks.append(lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False))
        try:
            booster = lgb.train(
                dict(self.params), train_set, num_boost_round=num_rounds,
                valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
            )
        finally:
            tracker.close()
        return LightGBMModel(booster)


class _RowBatches(xgb.DataIter):
    """Feeds X[rows] to a QuantileDMatrix in batches, so building a fold's
    matrix never materializes a full fancy-indexed copy (~12GB for the main
    fit set) on the host."""

    def __init__(self, X: np.ndarray, y: np.ndarray, rows: np.ndarray, batch_rows: int = 5_000_000):
        self._X, self._y, self._rows, self._batch_rows, self._pos = X, y, rows, batch_rows, 0
        super().__init__()

    def next(self, input_data) -> bool:
        if self._pos >= len(self._rows):
            return False
        r = self._rows[self._pos : self._pos + self._batch_rows]
        input_data(data=self._X[r], label=self._y[r], feature_names=FEATURE_COLUMNS)
        self._pos += self._batch_rows
        return True

    def reset(self) -> None:
        self._pos = 0


class XGBoostModel:
    def __init__(self, booster: xgb.Booster, n_rounds: int):
        self.booster = booster
        self.n_rounds = n_rounds

    def predict(self, X: np.ndarray, batch_rows: int = 2_000_000) -> np.ndarray:
        # X lives in host memory; predicting on the CUDA booster would make
        # XGBoost fall back to copying each batch into a DMatrix first.
        self.booster.set_param({"device": "cpu"})
        return np.concatenate(
            [
                self.booster.inplace_predict(X[s : s + batch_rows], iteration_range=(0, self.n_rounds))
                for s in range(0, len(X), batch_rows)
            ]
        ) if len(X) else np.array([], dtype=np.float32)

    def save(self, out_dir: Path) -> None:
        self.booster[: self.n_rounds].save_model(str(out_dir / "model.json"))

    def importance(self) -> pd.DataFrame:
        gain = self.booster.get_score(importance_type="total_gain")
        split = self.booster.get_score(importance_type="weight")
        return pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "gain": [gain.get(f, 0.0) for f in FEATURE_COLUMNS],
                "split": [split.get(f, 0.0) for f in FEATURE_COLUMNS],
            }
        ).sort_values("gain", ascending=False)


class XGBoostTrainer:
    name = "xgboost"
    params = config.XGB_PARAMS

    def __init__(self, data: TrainingData, device: str = "cuda"):
        self.X, self.y = data.X, data.y
        self.params = {**config.XGB_PARAMS, "device": device}

    def _matrix(self, rows: np.ndarray, ref: Optional[xgb.QuantileDMatrix] = None) -> xgb.QuantileDMatrix:
        return xgb.QuantileDMatrix(
            _RowBatches(self.X, self.y, rows), max_bin=self.params["max_bin"], ref=ref
        )

    def fit(
        self,
        fit_rows: np.ndarray,
        early_stop_rows: Optional[np.ndarray],
        num_rounds: int = config.MAX_ROUNDS,
        label: str = "training",
    ) -> XGBoostModel:
        t0 = time.time()
        dfit = self._matrix(fit_rows)
        evals = [(dfit, "fit")]
        if early_stop_rows is not None:
            evals.append((self._matrix(early_stop_rows, ref=dfit), "early_stop"))  # last = early-stopping set
        print(f"  built QuantileDMatrix for {len(fit_rows):,} fit rows in {time.time() - t0:.0f}s", flush=True)
        tracker = _RoundTracker(label, num_rounds, early_stopping=early_stop_rows is not None)
        try:
            booster = xgb.train(
                self.params, dfit, num_boost_round=num_rounds, evals=evals, verbose_eval=False,
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS if early_stop_rows is not None else None,
                callbacks=[_XGBoostProgress(tracker)],
            )
        finally:
            tracker.close()
        n_rounds = booster.best_iteration + 1 if early_stop_rows is not None else num_rounds
        return XGBoostModel(booster, n_rounds)


@dataclass
class EvalReport:
    threshold: float
    f05_by_country: pd.Series
    ceiling_by_country: pd.Series


def _tune_with_progress(
    data: TrainingData, scores: np.ndarray, rows: np.ndarray, eval_mask: np.ndarray, label: str
) -> ThresholdResult:
    bar: Optional[Progress] = None

    def on_step(done: int, total: int, best: float) -> None:
        nonlocal bar
        if bar is None:
            bar = Progress(total, label, "thresholds")
        bar.update(done, f"best macro F0.5 so far {best:.4f}")

    result = tune_threshold(
        data.entity[rows], scores, data.y[rows], data.truth["n_true"].to_numpy(), eval_mask, progress=on_step
    )
    if bar is not None:
        bar.close(f"best macro F0.5 {result.macro_f05:.4f} @ threshold {result.threshold:.4f}")
    return result


def evaluate_on(
    data: TrainingData,
    scores: np.ndarray,
    rows: np.ndarray,
    eval_mask: np.ndarray,
    threshold: Optional[float] = None,
    label: str = "threshold",
) -> EvalReport:
    """Macro F_0.5 on the `eval_mask` entities, using pairs `rows` scored by
    `scores`. Tunes the threshold on these same entities unless one is given.
    Also reports the candidate-set ceiling: F_0.5 of a perfect classifier on
    these candidates -- the gap between the two is the classifier's, the gap
    from the ceiling to 1.0 is blocking's."""
    entity, label_arr = data.entity[rows], data.y[rows]
    n_true = data.truth["n_true"].to_numpy()
    countries = data.truth["country"].to_numpy()
    if threshold is None:
        threshold = _tune_with_progress(data, scores, rows, eval_mask, label).threshold
    f05 = macro_f05_by_group(entity, scores >= threshold, label_arr, n_true, eval_mask, countries)
    ceiling = macro_f05_by_group(entity, label_arr.astype(bool), label_arr, n_true, eval_mask, countries)
    return EvalReport(threshold=threshold, f05_by_country=f05, ceiling_by_country=ceiling)


def _rows_for(data: TrainingData, entity_mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(entity_mask[data.entity])


def _subsample_entities(mask: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return mask
    rng = np.random.default_rng(seed)
    return mask & (rng.random(len(mask)) < frac)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def run_dev(data: TrainingData, trainer, out_dir: Path, fit_frac: float, loco: bool) -> None:
    countries = data.truth["country"].to_numpy()
    roles = split_entities(countries)
    roles[~data.scope] = -1  # out of scope: no role
    fit_mask = _subsample_entities(roles == ROLE_FIT, fit_frac, config.SPLIT_SEED)
    es_mask, val_mask = roles == ROLE_EARLY_STOP, roles == ROLE_VAL
    fit_rows, es_rows, val_rows = (_rows_for(data, m) for m in (fit_mask, es_mask, val_mask))
    print(
        f"split: fit {fit_mask.sum():,} entities / {len(fit_rows):,} pairs, "
        f"early-stop {es_mask.sum():,} / {len(es_rows):,}, dev_val {val_mask.sum():,} / {len(val_rows):,}"
    )

    _stage(f"training main model ({trainer.name}, up to {config.MAX_ROUNDS} rounds, early stopping)")
    t0 = time.time()
    model = trainer.fit(fit_rows, es_rows, label="main model")
    print(f"  kept {model.n_rounds} rounds, trained in {_fmt_duration(time.time() - t0)}", flush=True)

    _stage(f"scoring {len(val_rows):,} dev_val pairs + tuning threshold")
    val_scores = model.predict(data.X[val_rows])
    entity, label = data.entity[val_rows], data.y[val_rows]
    tuned = _tune_with_progress(data, val_scores, val_rows, val_mask, "threshold")
    report = evaluate_on(data, val_scores, val_rows, val_mask, threshold=tuned.threshold)

    summary = pd.DataFrame({"macro_f05": report.f05_by_country, "candidate_ceiling": report.ceiling_by_country})
    print(f"\n=== dev_val (threshold {tuned.threshold:.4f}) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    metrics = {
        "mode": "dev",
        "backend": trainer.name,
        "threshold": tuned.threshold,
        "best_iteration": model.n_rounds,
        "dev_val_macro_f05": report.f05_by_country.to_dict(),
        "dev_val_candidate_ceiling": report.ceiling_by_country.to_dict(),
        "n_fit_entities": int(fit_mask.sum()),
        "n_fit_pairs": int(len(fit_rows)),
        "n_val_entities": int(val_mask.sum()),
        "fit_entity_fraction": fit_frac,
        "params": trainer.params,
        "features": FEATURE_COLUMNS,
    }

    if loco:
        metrics["loco"] = run_loco(data, trainer, roles, fit_frac, main_report=report)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    tuned.curve.to_csv(out_dir / "threshold_curve.csv", index=False)
    model.importance().to_csv(out_dir / "feature_importance.csv", index=False)
    pd.DataFrame(
        {
            "source1_entity_id": data.truth["source1_entity_id"].to_numpy()[entity],
            "candidate_entity_id": data.candidate_ids.take(pa.array(val_rows)).to_numpy(),
            "score": val_scores.astype(np.float32),
            "label": label,
        }
    ).to_parquet(out_dir / "val_predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nartifacts written to {out_dir}")


def run_loco(data: TrainingData, trainer, roles: np.ndarray, fit_frac: float, main_report: EvalReport) -> dict:
    """Leave-one-country-out: for each train country C, train on every OTHER
    country only, tune the threshold on the other countries' dev_val (the
    only labels a truly unseen country would have), then score C's dev_val.

    Reported per held-out country:
      in_domain      main model (saw C), global threshold
      loco           model + threshold from other countries only -- France's
                     actual test-time situation
      loco_own_thr   same LOCO model, threshold re-tuned on C itself --
                     separates "the model degrades" from "the threshold
                     miscalibrates", i.e. whether a self-calibrating
                     threshold (plan.md) would recover the gap
    """
    countries = data.truth["country"].to_numpy()
    held_out_countries = np.unique(countries[roles >= 0])
    results = {}
    for i, held_out in enumerate(held_out_countries, start=1):
        others = countries != held_out
        fit_mask = _subsample_entities((roles == ROLE_FIT) & others, fit_frac, config.SPLIT_SEED)
        es_rows = _rows_for(data, (roles == ROLE_EARLY_STOP) & others)
        _stage(f"LOCO {i}/{len(held_out_countries)}: training without {held_out}")
        model = trainer.fit(_rows_for(data, fit_mask), es_rows, label=f"LOCO -{held_out}")

        other_val = (roles == ROLE_VAL) & others
        other_rows = _rows_for(data, other_val)
        thr = evaluate_on(
            data, model.predict(data.X[other_rows]), other_rows, other_val, label="threshold (others)"
        ).threshold

        held_val = (roles == ROLE_VAL) & ~others
        held_rows = _rows_for(data, held_val)
        held_scores = model.predict(data.X[held_rows])
        loco = evaluate_on(data, held_scores, held_rows, held_val, threshold=thr)
        own = evaluate_on(data, held_scores, held_rows, held_val, label=f"threshold ({held_out})")

        results[held_out] = {
            "in_domain": float(main_report.f05_by_country[held_out]),
            "loco": float(loco.f05_by_country[held_out]),
            "loco_own_thr": float(own.f05_by_country[held_out]),
            "loco_threshold": thr,
            "own_threshold": own.threshold,
            "candidate_ceiling": float(main_report.ceiling_by_country[held_out]),
        }

    table = pd.DataFrame(results).T
    table["loco_gap"] = table["in_domain"] - table["loco"]
    print("\n=== LOCO (dev_val of the held-out country) ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    return results


def run_final(data: TrainingData, trainer, out_dir: Path, dev_dir: Path) -> None:
    dev = json.loads((dev_dir / "metrics.json").read_text())
    dev_backend = dev.get("backend", "lightgbm")
    if dev_backend != trainer.name:
        raise SystemExit(
            f"dev run '{dev_dir.name}' used {dev_backend}; its round count and threshold don't transfer to "
            f"{trainer.name} -- rerun final with the same backend (--gpu iff the dev run used it)"
        )
    rounds, threshold = int(dev["best_iteration"]), float(dev["threshold"])
    _stage(f"final: retraining on all {len(data.y):,} pairs for {rounds} rounds, locked threshold {threshold:.4f}")
    t0 = time.time()
    model = trainer.fit(np.arange(len(data.y)), None, num_rounds=rounds, label="final model")
    print(f"  trained in {_fmt_duration(time.time() - t0)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    model.importance().to_csv(out_dir / "feature_importance.csv", index=False)
    metrics = {"mode": "final", "backend": trainer.name, "threshold": threshold, "num_rounds": rounds,
               "dev_run": dev_dir.name, "params": trainer.params, "features": FEATURE_COLUMNS}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"artifacts written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the pair classifier.")
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    parser.add_argument("--features", default="train", help="Directory name under data/processed/features/.")
    parser.add_argument("--run", default=None, help="Output directory name under data/processed/models/.")
    parser.add_argument("--fit-entity-frac", type=float, default=1.0,
                        help="dev: train on this fraction of fit entities (dev_val stays full).")
    parser.add_argument("--loco", action="store_true", help="dev: also run the leave-one-country-out test.")
    parser.add_argument("--dev-run", default="dev", help="final: dev run whose rounds/threshold to reuse.")
    parser.add_argument("--gpu", action="store_true",
                        help="Train XGBoost on CUDA instead of LightGBM on CPU (see the backend notes above).")
    args = parser.parse_args()
    run_name = args.run or args.mode

    run_start = time.time()
    _stage(f"loading ground truth + features/{args.features}")
    gt = load_ground_truth()
    truth = entity_truth(gt)
    labeler = PairLabeler(ground_truth_pairs(gt))
    data = load_training_data(config.FEATURES_DIR / args.features, truth, labeler, sampled=args.features != "train")
    print(f"  {len(data.y):,} pairs, {data.y.mean() * 100:.2f}% positive", flush=True)

    _stage("preparing " + ("XGBoost on CUDA" if args.gpu else "LightGBM on CPU (binning the dataset)"))
    t0 = time.time()
    trainer = XGBoostTrainer(data, device="cuda") if args.gpu else LightGBMTrainer(data)
    print(f"  ready in {time.time() - t0:.0f}s", flush=True)

    if args.mode == "dev":
        run_dev(data, trainer, config.MODELS_DIR / run_name, args.fit_entity_frac, args.loco)
    else:
        run_final(data, trainer, config.MODELS_DIR / run_name, config.MODELS_DIR / args.dev_run)
    _stage(f"done in {_fmt_duration(time.time() - run_start)}")


if __name__ == "__main__":
    main()
