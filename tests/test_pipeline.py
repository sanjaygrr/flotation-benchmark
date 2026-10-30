import json

import pandas as pd

from flotation_benchmark.benchmark import PROTOCOLS, Config, build_tables, feature_columns, run
from flotation_benchmark.cli import main
from flotation_benchmark.data import IRON_CONC

from .conftest import synthetic_raw


def _write_csv(df, path):
    out = df.drop(columns="sample").copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in out.columns[1:]:
        out[c] = out[c].map(lambda v: f"{v:.5f}".replace(".", ","))
    out.to_csv(path, index=False)


def test_iron_column_only_in_leaky_protocols(raw_synth):
    tables = build_tables(raw_synth)
    for p in PROTOCOLS:
        cols = feature_columns(p, tables[p.table])
        assert (IRON_CONC in cols) == p.use_iron
        if p.honest:
            assert IRON_CONC not in cols
            assert "silica_concentrate" not in cols


def test_forecast_features_exclude_target_hour(raw_synth):
    tables = build_tables(raw_synth)
    p = next(p for p in PROTOCOLS if p.key == "forecast")
    cols = feature_columns(p, tables["forecast"])
    assert all(c.endswith(("_lag0", "_lag1")) for c in cols)


def test_end_to_end_cli(tmp_path):
    raw = synthetic_raw(hours=300, per_hour=4)
    csv = tmp_path / "raw.csv"
    _write_csv(raw, csv)
    out = tmp_path / "results"
    assert main(["run", "--raw", str(csv), "--out", str(out), "--folds", "3",
                 "--gap-hours", "6", "--jobs", "1", "--models", "mean", "persistence", "ridge",
                 "lightgbm"]) == 0
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary.protocol) == {p.key for p in PROTOCOLS}
    assert summary[["mae_mean", "rmse_mean", "r2_mean"]].notna().all().all()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["data"]["gaps"][0]["missing_hours"] == 30
    assert (out / "importances.csv").exists()
    assert main(["report", "--results", str(out)]) == 0
    html = (out / "report.html").read_text()
    assert "Scores by protocol and model" in html and "<svg" in html
    assert "http://" not in html.replace("http://www.w3.org", "")


def test_quick_config_skips_importances(tmp_path, raw_synth):
    csv = tmp_path / "raw.csv"
    _write_csv(raw_synth, csv)
    out = run(csv, tmp_path / "r", Config(n_folds=2, gap_hours=2, importance_repeats=0, n_jobs=1),
              protocols=["walk_forward"], models=["mean", "ridge"])
    assert not (out / "importances.csv").exists()
