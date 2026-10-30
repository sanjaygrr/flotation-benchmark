"""Aggregation to the lab-assay timescale and data-quality checks."""

from __future__ import annotations

import pandas as pd

from .data import COLUMNS, IRON_CONC, TARGET, TIME

PROCESS_COLUMNS = [c for c in COLUMNS if c not in (IRON_CONC, TARGET)]


def aggregate_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 20 s rows to one row per hour.

    Process variables are averaged over the hour. The target is averaged too:
    in most hours it is a single repeated lab value, but in some hours the
    published file holds a linear ramp between two assays, and the mean of
    that ramp is its midpoint.

    Extra columns:
      n_samples            number of 20 s rows in the hour
      target_interpolated  True if the target varies inside the hour
      target_repeated      True if the target equals the previous hour's value
      assay_causal         the target with interpolated hours replaced by the
                           last real value before them; safe to use as an input
                           because it never depends on a later assay
    """
    g = raw.groupby(TIME, sort=True)
    hourly = g[COLUMNS].mean()
    hourly["n_samples"] = g.size().astype("int32")
    hourly["target_interpolated"] = g[TARGET].nunique() > 1
    prev = hourly[TARGET].shift(1)
    hourly["target_repeated"] = (hourly[TARGET] == prev) & ~hourly["target_interpolated"]
    hourly["assay_causal"] = hourly[TARGET].mask(hourly["target_interpolated"]).ffill()
    return hourly.reset_index()


def find_gaps(times: pd.Series, freq: str = "1h") -> pd.DataFrame:
    """List the holes in an hourly time index (consecutive stamps more than one step apart)."""
    t = pd.Series(pd.to_datetime(times)).sort_values().reset_index(drop=True)
    step = pd.Timedelta(freq)
    dt = t.diff()
    mask = dt > step
    gaps = pd.DataFrame(
        {
            "last_before": t.shift(1)[mask].values,
            "first_after": t[mask].values,
        }
    )
    gaps["missing_hours"] = ((gaps["first_after"] - gaps["last_before"]) / step - 1).astype(int)
    return gaps


def data_summary(raw: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    """Facts about the data that the report prints."""
    gaps = find_gaps(hourly[TIME])
    counts = hourly["n_samples"].value_counts().sort_index()
    feed_changes = int((hourly["iron_feed"].diff().fillna(1) != 0).sum())
    return {
        "rows_raw": int(len(raw)),
        "hours": int(len(hourly)),
        "start": str(hourly[TIME].min()),
        "end": str(hourly[TIME].max()),
        "samples_per_hour": {int(k): int(v) for k, v in counts.items()},
        "gaps": [
            {
                "last_before": str(r.last_before),
                "first_after": str(r.first_after),
                "missing_hours": int(r.missing_hours),
            }
            for r in gaps.itertuples()
        ],
        "hours_target_interpolated": int(hourly["target_interpolated"].sum()),
        "hours_target_repeated": int(hourly["target_repeated"].sum()),
        "feed_assay_changes": feed_changes,
        "corr_iron_silica_hourly": float(hourly[[TARGET, IRON_CONC]].corr().iloc[0, 1]),
        "corr_iron_silica_rows": float(raw[[TARGET, IRON_CONC]].corr().iloc[0, 1]),
        "target_lag1_autocorr_hourly": float(hourly[TARGET].autocorr(1)),
    }


def add_persistence(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add ``prev_target``: the last lab value from an earlier hour.

    This is what an operator already knows before the current assay comes
    back, so it is the natural naive predictor. It is taken from
    ``assay_causal``, so an interpolated hour (whose value depends on the next
    assay) is never used. If the previous available hour is across a gap it is
    still used (last known value).
    """
    out = hourly.copy()
    out["prev_target"] = out["assay_causal"].shift(1)
    return out


def make_forecast_table(hourly: pd.DataFrame, lags: int = 2) -> pd.DataFrame:
    """Build the one-hour-ahead forecasting table.

    For each hour t with a recorded hour t+1, the inputs are the process
    means of hours t, t-1, ..., t-lags+1 and the lab silica of hours t and
    t-1 (from ``assay_causal``, so no interpolated value that depends on a
    later assay is used). The output is the silica value of hour t+1. Rows whose lags or
    target cross a gap in the data are dropped, so every input is from the
    past of the target hour. ``date`` is the target hour.
    """
    h = hourly.set_index(TIME).asfreq("1h")
    feats = {}
    for k in range(lags):
        shifted = h[PROCESS_COLUMNS].shift(k)
        for c in PROCESS_COLUMNS:
            feats[f"{c}_lag{k}"] = shifted[c]
    feats["silica_lag0"] = h["assay_causal"]
    feats["silica_lag1"] = h["assay_causal"].shift(1)
    X = pd.DataFrame(feats, index=h.index)
    y = h[TARGET].shift(-1).rename("target_next")
    table = pd.concat([X, y], axis=1).dropna()
    table.index = table.index + pd.Timedelta("1h")
    table.index.name = TIME
    return table.reset_index()
