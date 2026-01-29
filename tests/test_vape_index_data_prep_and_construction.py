# -*- coding: utf-8 -*-
"""
Unit tests for my pipeline pieces

Tests the core workflow on a toy dataset:

- build fiscal year map

- prepare store df

- subset vaping

- construct index

- validate the index
"""

from src.vape_price_index import (
    build_fiscal_year_periods,
    prepare_store_df,
    subset_vaping_products,
    compute_vape_price_index_for_store,
)
from src.validation import validate_store_df, validate_vape_price_index_df
from tests.utils import make_tiny_test_df


# unit test for data prep
def test_prepare_store_df_and_validation():
    df = make_tiny_test_df()
    fy_map = build_fiscal_year_periods()
    df_prep = prepare_store_df(df, fy_map)
    # validate_store_df expects lowercase columns -> prepare_store_df already did that
    df_valid = validate_store_df(df_prep)

    assert "date" in df_valid.columns, "date column missing after preparation"
    assert "fiscal_year" in df_valid.columns, "fiscal_year column missing after preparation"
    assert df_valid["scan_type"].eq("GTIN").all(), "scan_type values incorrect after preparation"


# unit test for store-level price index construction
def test_vape_price_index_pipeline():
    df = make_tiny_test_df()
    fy_map = build_fiscal_year_periods()

    df_prep = prepare_store_df(df, fy_map)
    subcat = subset_vaping_products(df_prep)
    idx = compute_vape_price_index_for_store(subcat)

    # Validate structure
    idx_valid = validate_vape_price_index_df(idx)

    assert not idx_valid.empty
    assert "vape_price_index" in idx_valid.columns, "validate_vape_price_index_df() failed"
    assert "l_vape_price_index" in idx_valid.columns, "l_vape_price_index must be one of the columns"









