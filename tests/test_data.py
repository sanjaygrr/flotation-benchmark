import pandas as pd
import pytest

from flotation_benchmark.data import COLUMNS, canonical_name, parse_raw, read_raw


def test_canonical_names_match_both_sources():
    assert canonical_name("% Iron Feed") == "iron_feed"
    assert canonical_name("%_Iron_Feed") == "iron_feed"
    assert canonical_name("Ore Pulp pH") == "ore_pulp_ph"
    assert canonical_name("Flotation Column 07 Level") == "flotation_column_07_level"
    assert canonical_name("% Silica Concentrate") == "silica_concentrate"


def test_parse_decimal_comma_and_dates(sample_csv):
    df = read_raw(sample_csv)
    assert list(df.columns) == ["date", "sample", *COLUMNS]
    assert len(df) == 21
    assert df["date"].dtype.kind == "M"
    assert df.loc[0, "iron_feed"] == pytest.approx(55.2)
    assert df.loc[0, "silica_feed"] == pytest.approx(16.98)
    assert df.loc[0, "ore_pulp_density"] == pytest.approx(1.74)
    assert df.loc[0, "silica_concentrate"] == pytest.approx(1.31)
    assert all(df[c].dtype == "float64" for c in COLUMNS)


def test_sample_index_counts_rows_within_hour(sample_csv):
    df = read_raw(sample_csv)
    assert df.groupby("date")["sample"].apply(list).iloc[0] == [0, 1, 2]


def test_parse_rejects_unsorted_rows(sample_csv):
    raw = pd.read_csv(sample_csv, dtype=str).iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        parse_raw(raw)


def test_parse_rejects_bad_numbers(sample_csv):
    raw = pd.read_csv(sample_csv, dtype=str)
    raw.loc[0, "Starch Flow"] = "abc"
    with pytest.raises(ValueError):
        parse_raw(raw)


def test_parse_reports_missing_columns(sample_csv):
    raw = pd.read_csv(sample_csv, dtype=str).drop(columns=["% Silica Concentrate"])
    with pytest.raises(KeyError, match="silica_concentrate"):
        parse_raw(raw)
