import numpy as np
import pandas as pd
import pytest

from src.vape_price_index import compute_vape_price_index_for_store

@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "store_id": ["1", "1", "1"],
            "gtin": ["A", "A", "A"],
            "subcategory": ["Vaping Products"] * 3,
            "product_type": ["Pods"] * 3,
            "calendar_year": [2023, 2023, 2023],
            "calendar_month": [1, 2, 4],  # March missing
            "date": pd.PeriodIndex(["2023-01", "2023-02", "2023-04"], freq="M"), # non-consecutive months
            "fiscal_year": [2023, 2023, 2023],
            "unit_value_q": [10.0, 11.0, 15.0],
            "quantity": [100, 110, 150],
            "total_revenue_amount": [1000, 1210, 2250],
            "scan_type": ["GTIN"] * 3,
        }
    )

def test_consecutive_months_produce_valid_index(sample_data):
    """
    If months are consecutive, index values should not be NaN.    
    """
    index_df = compute_vape_price_index_for_store(
        sample_data,
        weight_basis="fiscal",
        index_kind="price",
    )

     # Feb should have a valid index
    feb_index = index_df.loc[index_df["date"] == pd.Period("2023-02", freq="M")]["vape_price_index"].iloc[0]

    assert not pd.isna(feb_index), "Consecutive months should produce valid index values for February."

def test_non_consecutive_months_produce_nan_index(sample_data):
    """
    If months are non-consecutive, index values for the missing months should be NaN.    
    """
    index_df = compute_vape_price_index_for_store(
        sample_data,
        weight_basis="fiscal",
        index_kind="price",
    )

    # April index should be NaN due to missing March data
    apr_index = index_df.loc[index_df["date"] == pd.Period("2023-04", freq="M")]["vape_price_index"].iloc[0]

    print(apr_index)
    
    assert pd.isna(apr_index), "Non-consecutive months should produce NaN index values for April."  





