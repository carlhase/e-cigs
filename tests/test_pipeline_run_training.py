from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.ml.pipeline as pipeline_mod

class DummyBestEstimator:
    """Minimal object that looks like a fitted sklearn classifier."""
    def __init__(self, probs: np.ndarray):
        # probs should be shape (n_samples, 2) for binary classification
        self._probs = probs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # ignore X, return predetermined probabilities
        return self._probs


class DummySearch:
    """Minimal object that looks like GridSearchCV/RandomizedSearchCV after fitting."""
    def __init__(self, best_estimator, best_params, best_score):
        self.best_estimator_ = best_estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.fit_called_with = None

    def fit(self, X, y):
        self.fit_called_with = (X, y)
        return self
    

@pytest.fixture
def cfg():
    """A mock configuration object that mimics the structure a YAML config file"""
    # SimpleNamespace is enough to mimic cfg.features.x_cols etc.
    return SimpleNamespace(
        features=SimpleNamespace(
            x_cols=["x1", "x2"],
            force_categorical=["x2"],
        ),
        data=SimpleNamespace(
            target_col="y",
        ),
        train=SimpleNamespace(
            test_size=0.2,
            random_state=123,
            stratify=True,
        ),
        # nested dicts to be passed to downstream functions
        preprocess={"dummy": "preprocess_cfg"},
        model={"name": "random_forest", "params": {"n_estimators": 10}},
        search={"method": "grid", "param_grid": {"model__max_depth": [None, 5]}, "cv": 3, "scoring": "roc_auc"},
    )

@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x1": [1, 2, 3, 4],
            "x2": ["a", "b", "a", "b"],
            "y": [0, 1, 0, 1],
        }
    )