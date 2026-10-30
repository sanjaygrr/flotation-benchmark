"""Evaluation protocols and the benchmark runner."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import __version__
from .data import IRON_CONC, TARGET, TIME, read_raw
from .metrics import score
from .models import ALL_MODELS, make_model
from .preprocess import (
    PROCESS_COLUMNS,
    add_persistence,
    aggregate_hourly,
    data_summary,
    make_forecast_table,
)
from .splits import random_kfold, time_holdout, walk_forward

FEED_COLUMNS = ("iron_feed", "silica_feed")


@dataclass(frozen=True)
class Protocol:
    key: str
    title: str
    table: str  # "rows" (20 s), "hourly" or "forecast"
    split: str  # "random", "holdout" or "walk_forward"
    use_iron: bool
    honest: bool
    description: str
    use_feed: bool = True


PROTOCOLS = [
    Protocol(
        "naive", "A. Naive", "rows", "random", True, False,
        "20 s rows, shuffled 5-fold, % Iron Concentrate as an input. "
        "Reproduces the common setup.",
    ),
    Protocol(
        "rows_random_no_iron", "A1. Naive without iron", "rows", "random", False, False,
        "As A, but without % Iron Concentrate. Isolates the effect of the iron column.",
    ),
    Protocol(
        "hourly_random_no_iron", "A2. Hourly, shuffled", "hourly", "random", False, False,
        "Hourly means, shuffled 5-fold, no iron. Isolates the effect of the repeated "
        "20 s rows.",
    ),
    Protocol(
        "time_holdout", "B. Time-ordered split", "hourly", "holdout", False, True,
        "Hourly means, first 80 % of hours for training, last 20 % for testing, no iron.",
    ),
    Protocol(
        "time_holdout_with_iron", "B1. Time-ordered, with iron", "hourly", "holdout", True, False,
        "As B, but with % Iron Concentrate. That column is a lab result of the same "
        "sample, so it is not available to a soft sensor.",
    ),
    Protocol(
        "walk_forward", "C. Walk-forward CV", "hourly", "walk_forward", False, True,
        "Hourly means, expanding-window CV over 5 consecutive test blocks with a gap "
        "between train and test, no iron.",
    ),
    Protocol(
        "walk_forward_no_feed", "C1. Walk-forward, no feed assays", "hourly", "walk_forward", False, True,
        "As C, but also without % Iron Feed and % Silica Feed. Those are lab assays too and their "
        "reporting time is not recorded.", use_feed=False,
    ),
    Protocol(
        "forecast", "D. One-hour-ahead forecast", "forecast", "walk_forward", False, True,
        "Predict the next hour's assay from process data of the current and previous "
        "hour and the last two lab values. Same CV as C.",
    ),
]
PROTOCOL_BY_KEY = {p.key: p for p in PROTOCOLS}


@dataclass
class Config:
    n_folds: int = 5
    gap_hours: int = 24
    test_frac: float = 0.2
    seed: int = 0
    n_jobs: int = 4
    importance_repeats: int = 5
    rows_stride: int = 1  # use every k-th 20 s row (1 = all); for quick runs


def build_tables(raw: pd.DataFrame, rows_stride: int = 1) -> dict[str, pd.DataFrame]:
    hourly = add_persistence(aggregate_hourly(raw))
    rows = raw.iloc[::rows_stride].reset_index(drop=True)
    rows = rows.merge(hourly[[TIME, "prev_target"]], on=TIME, how="left")
    forecast = make_forecast_table(hourly)
    forecast["prev_target"] = forecast["silica_lag0"]
    forecast = forecast.rename(columns={"target_next": TARGET})
    flags = ["target_interpolated", "target_repeated"]
    forecast = forecast.merge(hourly[[TIME, *flags]], on=TIME, how="left")
    return {"rows": rows, "hourly": hourly, "forecast": forecast}


def feature_columns(protocol: Protocol, table: pd.DataFrame) -> list[str]:
    if protocol.table == "forecast":
        skip = {TIME, TARGET, "prev_target", "target_interpolated", "target_repeated", "assay_causal"}
        return [c for c in table.columns if c not in skip]
    cols = list(PROCESS_COLUMNS)
    if not protocol.use_feed:
        cols = [c for c in cols if c not in FEED_COLUMNS]
    if protocol.use_iron:
        cols.append(IRON_CONC)
    return cols


def make_splits(protocol: Protocol, table: pd.DataFrame, cfg: Config):
    if protocol.split == "random":
        return random_kfold(len(table), cfg.n_folds, cfg.seed)
    if protocol.split == "holdout":
        return time_holdout(table[TIME], cfg.test_frac)
    if protocol.split == "walk_forward":
        return walk_forward(table[TIME], cfg.n_folds, pd.Timedelta(hours=cfg.gap_hours))
    raise ValueError(protocol.split)


def run_protocol(protocol: Protocol, table: pd.DataFrame, cfg: Config, models=ALL_MODELS):
    feats = feature_columns(protocol, table)
    X = table[feats].to_numpy(dtype=np.float32)
    y = table[TARGET].to_numpy(dtype=np.float64)
    prev = table["prev_target"].to_numpy(dtype=np.float64)
    times = table[TIME].to_numpy()

    scores, preds, imps = [], [], []
    for fold, (tr, te) in enumerate(make_splits(protocol, table, cfg)):
        fold_pred = pd.DataFrame({TIME: times[te], "fold": fold, "y_true": y[te]})
        for flag in ("target_interpolated", "target_repeated"):
            if flag in table.columns:
                fold_pred[flag] = table[flag].to_numpy()[te]
        for name in models:
            t0 = time.perf_counter()
            if name == "persistence":
                p = prev[te].copy()
                p[np.isnan(p)] = y[tr].mean()
            else:
                model = make_model(name, n_train=len(tr), seed=cfg.seed, n_jobs=cfg.n_jobs)
                model.fit(X[tr], y[tr])
                p = model.predict(X[te])
                if protocol.honest and name not in ("mean",) and cfg.importance_repeats > 0:
                    pi = permutation_importance(
                        model, X[te], y[te], scoring="neg_mean_absolute_error",
                        n_repeats=cfg.importance_repeats, random_state=cfg.seed,
                        n_jobs=1,
                    )
                    imps.append(pd.DataFrame({
                        "protocol": protocol.key, "model": name, "fold": fold,
                        "feature": feats, "importance": pi.importances_mean,
                    }))
            row = {
                "protocol": protocol.key, "model": name, "fold": fold,
                "n_train": len(tr), "n_test": len(te),
                "train_start": str(pd.Timestamp(times[tr].min())),
                "train_end": str(pd.Timestamp(times[tr].max())),
                "test_start": str(pd.Timestamp(times[te].min())),
                "test_end": str(pd.Timestamp(times[te].max())),
                **score(y[te], p),
                "seconds": round(time.perf_counter() - t0, 2),
            }
            scores.append(row)
            fold_pred[name] = p
            print(f"  {protocol.key:24s} fold {fold} {name:14s} "
                  f"MAE {row['mae']:.3f}  RMSE {row['rmse']:.3f}  R2 {row['r2']:+.3f}", flush=True)
        if protocol.table != "rows":
            preds.append(fold_pred)
    return (
        pd.DataFrame(scores),
        pd.concat(preds, ignore_index=True) if preds else None,
        pd.concat(imps, ignore_index=True) if imps else None,
    )


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    g = scores.groupby(["protocol", "model"], sort=False)
    out = g[["mae", "rmse", "r2"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_folds"] = g.size()
    return out.reset_index()


def run(raw_path: Path, out_dir: Path, cfg: Config | None = None,
        protocols: list[str] | None = None, models=ALL_MODELS) -> Path:
    cfg = cfg or Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw(raw_path)
    tables = build_tables(raw, cfg.rows_stride)
    summary = data_summary(raw, tables["hourly"])
    del raw

    all_scores, all_preds, all_imps = [], [], []
    for p in PROTOCOLS:
        if protocols and p.key not in protocols:
            continue
        print(f"[{p.title}]", flush=True)
        s, pr, im = run_protocol(p, tables[p.table], cfg, models)
        all_scores.append(s)
        if pr is not None:
            all_preds.append(pr.assign(protocol=p.key))
        if im is not None:
            all_imps.append(im)

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(out_dir / "scores.csv", index=False)
    summarize(scores).to_csv(out_dir / "summary.csv", index=False)
    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    if all_imps:
        pd.concat(all_imps, ignore_index=True).to_csv(out_dir / "importances.csv", index=False)

    source = Path(raw_path).parent / "SOURCE.json"
    meta = {
        "version": __version__,
        "config": asdict(cfg),
        "protocols": [asdict(p) for p in PROTOCOLS if not protocols or p.key in protocols],
        "models": list(models),
        "data": summary,
        "source": json.loads(source.read_text()) if source.exists() else None,
        "table_sizes": {k: int(len(v)) for k, v in tables.items()},
        "python": platform.python_version(),
        "packages": _versions(),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return out_dir


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for pkg in ["numpy", "pandas", "scikit-learn", "lightgbm"]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out
