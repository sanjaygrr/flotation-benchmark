"""Download and parse the raw flotation dataset.

The data are the public "Quality Prediction in a Mining Process" dataset
(iron ore reverse flotation plant, March to September 2017). We use the
OpenML mirror (dataset 43311) because it needs no account.

In the OpenML copy every column is stored as a string and numbers use a
decimal comma ("55,2"). The ``date`` column only has hourly resolution: the
roughly 180 rows that share a timestamp are the 20 s samples of that hour, in
file order.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_ID = 43311
SOURCE = {
    "name": "Quality Prediction in a Mining Process",
    "openml_id": OPENML_ID,
    "openml_page": f"https://www.openml.org/d/{OPENML_ID}",
    "url": "https://data.openml.org/datasets/0004/43311/dataset_43311.pq",
    "format": "parquet",
    # sha256 of the parquet file as served by OpenML on 2026-10-30.
    "sha256": "2f0675981a04e2747cfce0d5c16ea56d3f1d14817fcfcd7a96c469a020246f31",
    "license": "CC0: Public Domain (as declared on OpenML)",
    "original": "https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process",
}

TARGET = "silica_concentrate"
IRON_CONC = "iron_concentrate"
TIME = "date"

# Canonical column order after parsing.
COLUMNS = [
    "iron_feed",
    "silica_feed",
    "starch_flow",
    "amina_flow",
    "ore_pulp_flow",
    "ore_pulp_ph",
    "ore_pulp_density",
    *[f"flotation_column_{i:02d}_air_flow" for i in range(1, 8)],
    *[f"flotation_column_{i:02d}_level" for i in range(1, 8)],
    IRON_CONC,
    TARGET,
]

# Lab assays. Feed assays change slowly; the two concentrate assays are the
# hourly lab results (the target and its near mirror).
LAB_COLUMNS = ["iron_feed", "silica_feed", IRON_CONC, TARGET]


def canonical_name(name: str) -> str:
    """Map raw column names (Kaggle CSV or OpenML) to snake_case names."""
    s = name.strip().lower().replace("%", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(dest_dir: Path, force: bool = False) -> Path:
    """Download the raw parquet file into ``dest_dir`` and record its source."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "quality_prediction_mining.parquet"
    if out.exists() and not force:
        print(f"{out} already exists, skipping download (use --force to redo)")
    else:
        tmp = out.with_suffix(".part")
        print(f"downloading {SOURCE['url']}")
        with urllib.request.urlopen(SOURCE["url"], timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(out)
    digest = sha256sum(out)
    if SOURCE["sha256"] and digest != SOURCE["sha256"]:
        raise RuntimeError(
            f"checksum mismatch for {out}: got {digest}, expected {SOURCE['sha256']}"
        )
    meta = dict(SOURCE, sha256=digest, file=out.name, bytes=out.stat().st_size)
    (dest_dir / "SOURCE.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ok: {out} ({meta['bytes']} bytes, sha256 {digest})")
    return out


def _to_float(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        return col.astype("float64")
    s = col.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="raise").astype("float64")


def parse_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw table into typed columns.

    Returns a frame with ``date`` (hour stamp), ``sample`` (position of the
    row within its hour, 0-based, in file order) and the numeric columns in
    ``COLUMNS`` order.
    """
    df = df.rename(columns=canonical_name)
    missing = [c for c in [TIME, *COLUMNS] if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    out = pd.DataFrame({TIME: pd.to_datetime(df[TIME].astype("string").str.strip(),
                                             format="%Y-%m-%d %H:%M:%S")})
    for c in COLUMNS:
        out[c] = _to_float(df[c])
    if not out[TIME].is_monotonic_increasing:
        raise ValueError("raw rows are not in chronological order")
    out["sample"] = out.groupby(TIME).cumcount().astype("int32")
    out = out[[TIME, "sample", *COLUMNS]].reset_index(drop=True)
    float_cols = COLUMNS
    if out[float_cols].isna().any().any():
        bad = out[float_cols].isna().sum()
        raise ValueError(f"missing values after parsing: {bad[bad > 0].to_dict()}")
    return out


def read_raw(path: Path) -> pd.DataFrame:
    """Read the raw file (OpenML parquet or Kaggle CSV) and parse it."""
    path = Path(path)
    if path.suffix == ".parquet" or path.suffix == ".pq":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"unsupported file type: {path}")
    return parse_raw(df)


def float32_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast the numeric columns to float32 to save memory."""
    out = df.copy()
    out[COLUMNS] = out[COLUMNS].astype(np.float32)
    return out
