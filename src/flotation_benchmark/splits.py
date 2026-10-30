"""Train/test splitters.

Every splitter returns a list of ``(train_idx, test_idx)`` pairs of integer
positions into the table it was given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_kfold(n: int, n_folds: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled k-fold, ignoring time. This is the leaky setup."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    out = []
    for i in range(n_folds):
        test = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(n_folds) if j != i]))
        out.append((train, test))
    return out


def time_holdout(times: pd.Series, test_frac: float = 0.2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Single chronological split: the last ``test_frac`` of the hours is the test set."""
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    if not (np.diff(t.astype("int64")) >= 0).all():
        raise ValueError("times must be sorted")
    uniq = np.unique(t)
    cut = uniq[int(np.floor(len(uniq) * (1 - test_frac)))]
    train = np.flatnonzero(t < cut)
    test = np.flatnonzero(t >= cut)
    return [(train, test)]


def walk_forward(
    times: pd.Series, n_folds: int = 5, gap: str | pd.Timedelta = "24h"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window blocked CV with an embargo between train and test.

    The sorted unique time stamps are cut into ``n_folds + 1`` contiguous
    blocks. Fold ``i`` tests on block ``i + 1`` and trains on everything that
    ends at least ``gap`` before the first test stamp.
    """
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    if not (np.diff(t.astype("int64")) >= 0).all():
        raise ValueError("times must be sorted")
    gap = pd.Timedelta(gap).to_timedelta64()
    uniq = np.unique(t)
    blocks = np.array_split(uniq, n_folds + 1)
    out = []
    for i in range(1, n_folds + 1):
        test_start, test_end = blocks[i][0], blocks[i][-1]
        train = np.flatnonzero(t < test_start - gap)
        test = np.flatnonzero((t >= test_start) & (t <= test_end))
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"fold {i} is empty; use fewer folds or a smaller gap")
        out.append((train, test))
    return out


def max_train_time_before_test(times: pd.Series, train: np.ndarray, test: np.ndarray) -> pd.Timedelta:
    """Time between the last training stamp and the first test stamp (negative if they overlap)."""
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    return pd.Timedelta(t[test].min() - t[train].max())
