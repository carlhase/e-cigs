from __future__ import annotations

import pandas as pd
import pytest

from src.ml.features import make_xy


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": ["x", "y", "z"],
            "target": [0, 1, 0],
        }
    )

def test_make_xy_happy_path_returns_expected_types_and_shapes(sample_df):
    X, y = make_xy(
        sample_df, 
        x_cols=["a", "c"], 
        y_col="target"
        )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_make_xy_preserves_column_order(sample_df):
    # Intentionally non-alphabetical order
    x_cols = ["c", "a"]
    X, _ = make_xy(sample_df, x_cols=x_cols, y_col="target")

    assert list(X.columns) == x_cols


def test_make_xy_raises_keyerror_for_missing_x_cols(sample_df):
    with pytest.raises(KeyError, match=r"Missing X columns in df:"):
        make_xy(sample_df, x_cols=["a", "does_not_exist"], y_col="target")


def test_make_xy_missing_x_cols_error_message_contains_sorted_missing_cols(sample_df):
    # Missing columns should be sorted in the message: ['m1', 'm2']
    with pytest.raises(KeyError, match=r"Missing X columns in df: \['m1', 'm2'\]"):
        make_xy(sample_df, x_cols=["a", "m2", "m1"], y_col="target")


def test_make_xy_raises_keyerror_for_missing_y_col(sample_df):
    with pytest.raises(KeyError, match=r"Missing target column 'bad_target' in df\."):
        make_xy(sample_df, x_cols=["a", "b"], y_col="bad_target")


def test_make_xy_returns_copies_not_views(sample_df):
    X, y = make_xy(sample_df, x_cols=["a", "b"], y_col="target")

    # Mutating returned X/y should not mutate original df
    X.loc[0, "a"] = 999
    y.iloc[0] = 999

    assert sample_df.loc[0, "a"] == 1
    assert sample_df.loc[0, "target"] == 0


def test_make_xy_allows_duplicate_x_cols_and_preserves_them(sample_df):
    # pandas allows duplicate column selection; resulting df will have duplicate labels
    X, _ = make_xy(sample_df, x_cols=["a", "a", "b"], y_col="target")

    assert list(X.columns) == ["a", "a", "b"]
    assert X.shape == (3, 3)