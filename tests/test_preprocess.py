import numpy as np
import pandas as pd
import pytest

from flotation_benchmark.data import TARGET, read_raw
from flotation_benchmark.preprocess import (
    PROCESS_COLUMNS,
    add_persistence,
    aggregate_hourly,
    data_summary,
    find_gaps,
    make_forecast_table,
)


@pytest.fixture
def hourly(sample_csv):
    return aggregate_hourly(read_raw(sample_csv))


def test_one_row_per_hour(sample_csv, hourly):
    raw = read_raw(sample_csv)
    assert len(hourly) == raw["date"].nunique() == 7
    assert hourly["date"].is_unique
    assert (hourly["n_samples"] == 3).all()


def test_process_means_and_target(sample_csv, hourly):
    raw = read_raw(sample_csv)
    first = raw[raw.date == raw.date.min()]
    assert hourly.loc[0, "starch_flow"] == pytest.approx(first["starch_flow"].mean())
    assert hourly.loc[0, TARGET] == pytest.approx(1.31)


def test_interpolated_and_repeated_flags(hourly):
    h = hourly.set_index("date")
    assert h.loc["2017-03-10 09:00", "target_interpolated"]
    assert h.loc["2017-03-10 09:00", TARGET] == pytest.approx(1.8)  # midpoint of 1.9, 1.8, 1.7
    assert h.loc["2017-03-10 02:00", "target_repeated"]
    assert not h.loc["2017-03-10 03:00", "target_repeated"]
    assert h["target_interpolated"].sum() == 1


def test_find_gaps(hourly):
    gaps = find_gaps(hourly["date"])
    assert len(gaps) == 1
    g = gaps.iloc[0]
    assert g.last_before == pd.Timestamp("2017-03-10 04:00")
    assert g.first_after == pd.Timestamp("2017-03-10 08:00")
    assert g.missing_hours == 3


def test_data_summary(sample_csv, hourly):
    s = data_summary(read_raw(sample_csv), hourly)
    assert s["rows_raw"] == 21 and s["hours"] == 7
    assert s["gaps"][0]["missing_hours"] == 3
    assert s["hours_target_interpolated"] == 1


def test_persistence_uses_previous_hour_only(hourly):
    h = add_persistence(hourly)
    assert np.isnan(h.loc[0, "prev_target"])
    assert (h["prev_target"].iloc[1:].to_numpy() == h["assay_causal"].iloc[:-1].to_numpy()).all()


def test_causal_assay_never_uses_interpolated_values(hourly):
    h = add_persistence(hourly).set_index("date")
    # 09:00 is a ramp towards the next assay; the causal series keeps the 08:00 value
    assert h.loc["2017-03-10 09:00", "assay_causal"] == pytest.approx(2.0)
    assert h.loc["2017-03-10 10:00", "prev_target"] == pytest.approx(2.0)
    assert h.loc["2017-03-10 10:00", "assay_causal"] == pytest.approx(1.8)


def test_forecast_table_uses_only_past(raw_synth):
    hourly = aggregate_hourly(raw_synth)
    tab = make_forecast_table(hourly, lags=2)
    h = hourly.set_index("date")
    # target is the assay of the stamped hour; inputs come from the two hours before it
    for _, r in tab.sample(20, random_state=0).iterrows():
        t = r["date"]
        assert r["target_next"] == pytest.approx(h.loc[t, TARGET])
        assert r["silica_lag0"] == pytest.approx(h.loc[t - pd.Timedelta("1h"), "assay_causal"])
        assert r["silica_lag1"] == pytest.approx(h.loc[t - pd.Timedelta("2h"), "assay_causal"])
        c = PROCESS_COLUMNS[3]
        assert r[f"{c}_lag0"] == pytest.approx(h.loc[t - pd.Timedelta("1h"), c])
        assert r[f"{c}_lag1"] == pytest.approx(h.loc[t - pd.Timedelta("2h"), c])


def test_forecast_table_skips_rows_across_gaps(raw_synth):
    hourly = aggregate_hourly(raw_synth)
    tab = make_forecast_table(hourly, lags=2)
    present = set(hourly["date"])
    for t in tab["date"]:
        for k in (1, 2):
            assert t - pd.Timedelta(hours=k) in present
