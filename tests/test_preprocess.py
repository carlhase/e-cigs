import numpy as np
import pandas as pd
import pytest

from src.ml.preprocess import (
    split_num_cat_columns,
    build_preprocessor_from_cfg,
)

"""
Remember: 
- A fixture is a function that prepares data or state for tests, 
and pytest automatically injects it where needed. 
- Each test gets its own fresh copy of the fixture
"""
# helper functions for unit tests
@pytest.fixture
def tiny_df():
    """
    Small dataframe with:
      - numeric cols: x1, x2
      - categorical cols: city
      - int-coded category: division (we will force it categorical)
      - missing values in numeric and categorical for imputer tests

      - Whenever a test asks for tiny_df, this function is executed 
        and the result is given to the test.
    """
    return pd.DataFrame(
        {
            "x1": [1.0, np.nan, 3.0],   # numeric col
            "x2": [10.0, 11.0, np.nan],
            "city": ["Berlin", None, "Munich"], # cat col
            "division": [1, 2, 2],  # int but should be treated as categorical
        }
    )

@pytest.fixture
def preprocess_cfg_default():
    """
    Sample preprocessing configuration mimicking YAML structure.
    mirrors conf/preprocess/default.yaml
    """
    return {
        "numerical": {
            "imputer": {
                "strategy": "mean",
                "fill_value": None,
            },
            "scaler": {
                "kind": "standard",
            },
        },
        "categorical": {
            "imputer": {
                "strategy": "most_frequent",
                "fill_value": None,
            },
            "onehot": {
                "handle_unknown": "ignore", 
                "sparse_output": False
                },
        }
    }

@pytest.fixture
def preprocess_cfg_no_impute():
    """mirrors conf/preprocess/no_impute.yaml"""
    return {
        "numerical": {
            "imputer": None,
            "scaler": {"kind": "none"},
        },
        "categorical": {
            "imputer": None,
            "onehot": {
                "handle_unknown": "ignore", 
                "sparse_output": False
                },
        },
    }

# tests
def test_split_num_cat_columns_basic(tiny_df: pd.DataFrame):
    """ Test basic functionality of split_num_cat_columns."""
    num_cols, cat_cols = split_num_cat_columns(tiny_df)

    assert set(num_cols) == {"x1", "x2", "division"}
    assert set(cat_cols) == {"city"}


def test_split_num_cat_columns_force_categorical(tiny_df: pd.DataFrame):
    num_cols, cat_cols = split_num_cat_columns(tiny_df, force_categorical=["division"])

    assert "division" not in num_cols, "Division should not be in the numerical columns list"
    assert "division" in cat_cols, "Division should be in the category columns list"
    assert set(num_cols) == {"x1", "x2"}, "Numerical columns list is wrong"
    assert set(cat_cols) == {"city", "division"}, "Categorical columns list is wrong"


def test_build_preprocessor_default_runs_and_transforms(
    tiny_df,
    preprocess_cfg_default):
    """Tests the build_preprocessor_from_cfg function with default config"""
    # split columns as orchestration layer would
    num_cols, cat_cols = split_num_cat_columns(tiny_df, force_categorical=["division"])

    # build preprocessing pipeline as orchestration layer would
    pre = build_preprocessor_from_cfg(num_cols, cat_cols, preprocess_cfg_default)

    # Create Xt, the transformed feature matrix
    Xt = pre.fit_transform(tiny_df) 
    """ Note: fit_transform() is implicitly imported via 
    src.preprocess.build_preprocessor_from_cfg (via ColumnTransformer)
    """

    # should return a numeric matrix (numpy array) with finite values
    assert hasattr(Xt, "shape"), "The matrix Xt should have shape"
    assert Xt.shape[0] == len(tiny_df), (
        "The number of rows in Xt should equal the no. rows in tiny_df"
        )
    assert np.isfinite(Xt).all(), "Xt should have finite values"
    assert not np.isnan(Xt).any(), (
        "With median imputation and scaling, numeric part should have no NaNs"
    )


def test_no_impute_leaves_missing_numeric_as_nan(
    tiny_df,
    preprocess_cfg_no_impute    # preprocess config with no impute
    ):
    # split columns as orchestration layer would
    num_cols, cat_cols = split_num_cat_columns(tiny_df, force_categorical=["division"])

    # build preprocessing pipeline as orchestration layer would
    pre = build_preprocessor_from_cfg(num_cols, cat_cols, preprocess_cfg_no_impute)

    # create the transformed feature matrix
    Xt = pre.fit_transform(tiny_df)

    # No numeric imputation + no scaling means numeric NaNs remain
    # OneHotEncoder doesn't introduce NaNs, so this should still contain NaNs overall.
    assert np.isnan(Xt).any()


def test_unknown_scaler_kind(tiny_df, preprocess_cfg_default):
    """
    Test that if I override the default scaler with an unknown scaler kind,
    an error is raised
    """
    num_cols, cat_cols = split_num_cat_columns(tiny_df, force_categorical=["division"])

    # create new preprocess config with a bad scaler
    bad_cfg = dict(preprocess_cfg_default)
    bad_cfg["numerical"] = dict(preprocess_cfg_default["numerical"])
    bad_cfg["numerical"]["scaler"] = {"kind": "not_a_scaler"}

    # pytest assertion that a specific error is raised
    with pytest.raises(ValueError, match="Unsupported scaler kind:"):
        _=build_preprocessor_from_cfg(num_cols, cat_cols, bad_cfg)    


def test_onehot_handle_unknown_ignore_does_not_fail(preprocess_cfg_default):
    """
    Test to ensures that the preprocessing pipeline does not crash when it sees 
    unseen categorical values at prediction time.
    """
    # Fit on data without "Hamburg", then transform with new category -> should not error
    train = pd.DataFrame(
        {"x1": [1.0, 2.0], "x2": [10.0, 11.0], "city": ["Berlin", "Munich"], "division": [1, 2]}
    )
    test = pd.DataFrame(
        {"x1": [3.0], "x2": [12.0], "city": ["Hamburg"], "division": [2]}
    )

    num_cols, cat_cols = split_num_cat_columns(train, force_categorical=["division"])
    
    # build preprocessing pipeline as orchestration layer would
    pre = build_preprocessor_from_cfg(num_cols, cat_cols, preprocess_cfg_default)

    # learn and apply imputation/scaling/encoding parameters on two different datasets
    pre.fit(train)              # .fit() learns imputation/scaling/encoding parameters
    Xt = pre.transform(test)    # .transform() applies the parameters learned from fit

    assert Xt.shape[0] == 1