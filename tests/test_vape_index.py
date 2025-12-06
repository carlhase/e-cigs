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
from src.validation import validate_store_df, validate_vape_index_df
from tests.utils import make_tiny_test_df


# unit test for data prep
def test_prepare_store_df_and_validation():
    df = make_tiny_test_df()
    fy_map = build_fiscal_year_periods()

    df_prep = prepare_store_df(df, fy_map)
    # validate_store_df expects lowercase columns -> prepare_store_df already did that
    df_valid = validate_store_df(df_prep)

    assert "date" in df_valid.columns
    assert "fiscal_year" in df_valid.columns
    assert df_valid["scan_type"].eq("GTIN").all()


# unit test for store-level price index construction
def test_vape_price_index_pipeline():
    df = make_tiny_test_df()
    fy_map = build_fiscal_year_periods()

    df_prep = prepare_store_df(df, fy_map)
    subcat = subset_vaping_products(df_prep)
    idx = compute_vape_price_index_for_store(subcat)

    # Validate structure
    idx_valid = validate_vape_index_df(idx)

    assert not idx_valid.empty
    assert "vape_price_index" in idx_valid.columns
    assert "l_vape_price_index" in idx_valid.columns


# unit test for store-level price index construction, using calendar year weights
def test_vape_price_index_pipeline_calendar_year_weights():
    """
    Same tiny pipeline as the fiscal-year test, but using calendar-year
    weights instead of fiscal-year weights.

    Recall: compute_vape_price_index_for_store has a weight_basis
    parameter: "fiscal" (default) or "calendar".
    """
    df = make_tiny_test_df()
    fy_map = build_fiscal_year_periods()

    # Optional: tweak the tiny DF so it spans at least two calendar years,
    # to make calendar-year grouping meaningful.
    # For example, bump the last row into the next year:
    df.loc[df.index[-1], "CALENDAR_YEAR"] = df["CALENDAR_YEAR"].max() + 1

    df_prep = prepare_store_df(df, fy_map)
    df_valid = validate_store_df(df_prep)

    subcat = subset_vaping_products(df_valid)

    # Use calendar-year weights instead of fiscal-year weights
    idx_cal = compute_vape_price_index_for_store(
        subcat,
        weight_basis="calendar",
    )

    # Validate structure and basic sanity via the same schema
    idx_cal_valid = validate_vape_index_df(idx_cal)

    assert not idx_cal_valid.empty
    assert "vape_price_index" in idx_cal_valid.columns
    assert "l_vape_price_index" in idx_cal_valid.columns

    # Extra sanity: no infinities in the log index
    assert (idx_cal_valid["l_vape_price_index"].replace([float("inf"), float("-inf")], float("nan")).notna().all()
            or idx_cal_valid["l_vape_price_index"].isna().any())









