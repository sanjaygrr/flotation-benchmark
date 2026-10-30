import pytest

from flotation_benchmark.cli import main


def test_run_rejects_unknown_protocol(tmp_path, capsys):
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        main(["run", "--raw", str(raw), "--protocols", "nope"])
    assert exc.value.code == 2
    assert "unknown protocol" in capsys.readouterr().err


def test_run_rejects_unknown_model(tmp_path, capsys):
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["run", "--raw", str(raw), "--models", "nope"])
    assert "unknown model" in capsys.readouterr().err


def test_run_reports_corrupt_file(tmp_path, capsys):
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"garbage")
    with pytest.raises(SystemExit):
        main(["run", "--raw", str(raw), "--out", str(tmp_path / "out")])
    assert "could not read" in capsys.readouterr().err


def test_report_without_results(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["report", "--results", str(tmp_path / "missing")])
    assert "no results" in capsys.readouterr().err


def test_run_reports_wrong_format(tmp_path, capsys):
    raw = tmp_path / "raw.csv"
    raw.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        main(["run", "--raw", str(raw), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert "not in the expected format" in err
    assert "fetch --force" not in err
