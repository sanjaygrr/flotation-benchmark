import numpy as np
import pandas as pd
import pytest

from flotation_benchmark.preprocess import aggregate_hourly
from flotation_benchmark.splits import random_kfold, time_holdout, walk_forward


@pytest.fixture
def times(raw_synth):
    return aggregate_hourly(raw_synth)["date"]


def test_random_kfold_partitions(times):
    n = len(times)
    folds = random_kfold(n, 5, seed=1)
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests) == list(range(n))
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == n


def test_time_holdout_has_no_overlap(times):
    (tr, te), = time_holdout(times, 0.2)
    t = times.to_numpy()
    assert t[tr].max() < t[te].min()
    assert len(tr) + len(te) == len(times)
    assert abs(len(te) / len(times) - 0.2) < 0.01


@pytest.mark.parametrize("gap_h", [0, 6, 24])
def test_walk_forward_no_test_time_inside_train_range(times, gap_h):
    t = times.to_numpy()
    folds = walk_forward(times, n_folds=4, gap=pd.Timedelta(hours=gap_h))
    assert len(folds) == 4
    prev_test_end = None
    for tr, te in folds:
        # every test stamp is later than every train stamp, by at least the gap
        assert (t[te].min() - t[tr].max()) > pd.Timedelta(hours=gap_h)
        assert not ((t[te] >= t[tr].min()) & (t[te] <= t[tr].max())).any()
        if prev_test_end is not None:
            assert t[te].min() > prev_test_end
        prev_test_end = t[te].max()


def test_walk_forward_expands(times):
    folds = walk_forward(times, n_folds=4, gap="0h")
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_walk_forward_on_repeated_stamps_keeps_hours_together(raw_synth):
    t = raw_synth["date"]
    for tr, te in walk_forward(t, n_folds=3, gap="2h"):
        assert not set(t.iloc[tr]) & set(t.iloc[te])


def test_splitters_reject_unsorted(times):
    shuffled = times.sample(frac=1, random_state=0)
    with pytest.raises(ValueError):
        walk_forward(shuffled, 3)
    with pytest.raises(ValueError):
        time_holdout(shuffled)


def test_random_rows_split_leaks_hours(raw_synth):
    # The naive split puts rows of the same hour on both sides; this is the leak we measure.
    t = raw_synth["date"].to_numpy()
    tr, te = random_kfold(len(t), 5, seed=0)[0]
    assert set(t[tr]) & set(t[te])
