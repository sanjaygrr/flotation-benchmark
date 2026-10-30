from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flotation_benchmark.data import COLUMNS, TARGET, TIME

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv() -> Path:
    return FIXTURES / "sample_raw.csv"


def synthetic_raw(hours: int = 400, per_hour: int = 6, seed: int = 0,
                  gap: tuple[int, int] | None = (100, 130)) -> pd.DataFrame:
    """A parsed-looking raw table with a slow AR(1) target and a hole in time."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2017-03-10 01:00", periods=hours, freq="h")
    if gap:
        stamps = stamps.delete(slice(*gap))
    n_h = len(stamps)
    y = np.empty(n_h)
    y[0] = 2.0
    for i in range(1, n_h):
        y[i] = 2.0 + 0.8 * (y[i - 1] - 2.0) + rng.normal(scale=0.3)
    rows = {TIME: np.repeat(stamps.values, per_hour)}
    for c in COLUMNS:
        rows[c] = rng.normal(size=n_h * per_hour)
    rows[TARGET] = np.repeat(y, per_hour)
    # Make one process variable informative.
    rows["amina_flow"] = rows[TARGET] + rng.normal(scale=0.5, size=n_h * per_hour)
    df = pd.DataFrame(rows)
    df["sample"] = df.groupby(TIME).cumcount()
    return df[[TIME, "sample", *COLUMNS]]


@pytest.fixture
def raw_synth() -> pd.DataFrame:
    return synthetic_raw()
