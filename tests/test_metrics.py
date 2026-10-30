import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from flotation_benchmark.metrics import mae, r2, rmse, score


def test_against_sklearn():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=0.5, size=200)
    assert mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))
    assert r2(y, p) == pytest.approx(r2_score(y, p))


def test_known_values():
    y = [1.0, 2.0, 3.0]
    assert score(y, y) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
    assert mae(y, [2.0, 2.0, 2.0]) == pytest.approx(2 / 3)
    assert rmse(y, [2.0, 2.0, 2.0]) == pytest.approx(np.sqrt(2 / 3))
    assert r2(y, [2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert r2(y, [3.0, 2.0, 1.0]) == pytest.approx(-3.0)


def test_input_checks():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        rmse([], [])
    with pytest.raises(ValueError):
        r2([1.0, np.nan], [1.0, 2.0])
    assert np.isnan(r2([1.0, 1.0], [1.0, 2.0]))
