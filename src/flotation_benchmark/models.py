"""Model zoo. Settings are deliberately light so the benchmark runs on a laptop."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINES = ["mean", "persistence"]
LEARNED = ["ridge", "random_forest", "lightgbm"]
ALL_MODELS = BASELINES + LEARNED


def make_model(name: str, n_train: int, seed: int = 0, n_jobs: int = 4):
    """Return an unfitted estimator. ``persistence`` is handled by the runner."""
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            max_features=0.33,
            # Cap the bootstrap size so the 20 s protocols stay within memory.
            max_samples=min(n_train, 50_000),
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"unknown model: {name}")
