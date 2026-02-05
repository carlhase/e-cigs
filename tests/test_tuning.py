"""
Unit tests the build_search() function in src/ml/tuning.py
Tests:
    1. The right class is returned (RandomizedSearchCV vs GridSearchCV)
    2. The right attributes/parameters are wired in (cv, scoring, n_jobs, etc.)
    3. The right grid key is used (param_distributions vs param_grid)
    4. Missing/unknown methods raise a clear error
    5. Optional defaults (e.g. verbose, n_jobs) behave as intended

Note: this module does not actually fit anything. It justs asserts.
"""

import pytest

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.ml.tuning import build_search


@pytest.fixture
def dummy_pipe() -> Pipeline:
    # Any sklearn estimator pipeline is fine; I won't actually fit anything.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression())
        ]
    )


def test_build_search_randomized_returns_randomized_search(dummy_pipe):
    """
    Test that RandomizedSearchCV class is returned and the right 
    attributes/parameters are wired in
    """
    # mimick a randomized config from a yaml file
    cfg = {
        "method": "randomized",
        "n_iter": 10,
        "cv": 5,
        "scoring": "roc_auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": 2,
        "param_distributions": {
            "model__C": [0.1, 1.0, 10.0],
        },
    }

    search = build_search(dummy_pipe, cfg)

    assert isinstance(search, RandomizedSearchCV)
    assert search.estimator is dummy_pipe
    assert search.n_iter == 10
    assert search.cv == 5
    assert search.scoring == "roc_auc"
    assert search.n_jobs == -1
    assert search.verbose == 2
    assert search.param_distributions == {"model__C": [0.1, 1.0, 10.0]}


def test_built_search_grid_returns_grid_search(dummy_pipe):
    """
    Test that GridSearchCV class is returned and the right
    attributes/parameters are wired in
    """
    # mimick a grid search config from a yaml file
    cfg = {
        "method": "grid",
        "cv": 3,
        "scoring": "accuracy",
        "n_jobs": 1,
        "verbose": 0,
        "param_grid": {
            "model__C": [0.1, 1.0],
            "model__penalty": ["l2"],
        },
    }

    search = build_search(dummy_pipe, cfg)

    assert isinstance(search, GridSearchCV)
    assert search.estimator is dummy_pipe
    assert search.cv == 3
    assert search.scoring == "accuracy"
    assert search.param_grid == {
        "model__C": [0.1, 1.0],
        "model__penalty": ["l2"]
    }


def test_build_search_unknown_method_raises_value_error(dummy_pipe):
    cfg = {
        "method": "bayesopt",
        "cv": 3,
        "scoring": "roc_auc",
    }

    with pytest.raises(ValueError, match = r"Unsupported search method: bayesopt"):
        build_search(dummy_pipe, cfg)


def test_build_search_default_verbose_and_n_jobs(dummy_pipe):
    # omitting optional keys should fall back to defaults
    cfg = {
        "method": "grid",
        "cv": 5,
        "scoring": "roc_auc",
        "param_grid": {"model__C": [1.0]},
    }

    search = build_search(dummy_pipe, cfg)

    assert isinstance(search, GridSearchCV)
    assert search.verbose == 0
    assert search.n_jobs == -1
