# -*- coding: utf-8 -*-
"""
Provides a small DataFrame that mimics my raw store data, 
so I can test the pipeline quickly.
"""

# tests/utils.py
import numpy as np
import pandas as pd


def make_tiny_test_df() -> pd.DataFrame:
    """
    Tiny synthetic dataset for a single store/GTIN over 3 months.
    Just enough columns to drive the vape_price_index pipeline.
    """
    return pd.DataFrame({
        # core identifiers
        "STORE_ID": [1001, 1001, 1001],
        "CALENDAR_YEAR": [2022, 2022, 2022],
        "CALENDAR_MONTH": [10, 11, 12],

        # product id
        "GTIN": ["12345", "12345", "12345"],

        # nonscan info (can be null)
        "NONSCAN_CATEGORY": [np.nan, np.nan, np.nan],
        "NONSCAN_SUBCATEGORY": [np.nan, np.nan, np.nan],
        "NONSCAN_DETAIL": [np.nan, np.nan, np.nan],

        # quantities & transactions
        "QUANTITY": [10.0, 11.0, 12.0],
        "QUANTITY_WITH_DISCOUNT": [5.0, 5.5, 6.0],
        "TRANSACTION_COUNT": [1, 1, 1],
        "TRANSACTION_COUNT_WITH_DISCOUNT": [1, 1, 1],

        # revenue
        "TOTAL_REVENUE_AMOUNT": [100.0, 110.0, 120.0],

        # scan type
        "SCAN_TYPE": ["GTIN", "GTIN", "GTIN"],

        # category info
        "CATEGORY": ["MERCHANDISE", "MERCHANDISE", "MERCHANDISE"],
        "SUBCATEGORY": ["Vaping Products", "Vaping Products", "Vaping Products"],
        "MANUFACTURER": [np.nan, np.nan, np.nan],
        "BRAND": [np.nan, np.nan, np.nan],
        "PRODUCT_TYPE": ["Disposable", "Disposable", "Disposable"],
        "SUB_PRODUCT_TYPE": [np.nan, np.nan, np.nan],
        "UNIT_SIZE": [np.nan, np.nan, np.nan],
        "PACK_SIZE": [np.nan, np.nan, np.nan],
        "PRODUCT_DESCRIPTION": [np.nan, np.nan, np.nan],

        # unit values
        "Q_PLUS_QWD": [10.0, 11.0, 12.0],
        "UNIT_VALUE_Q_PLUS_QWD": [3.0, 3.5, 3.8],
        "UNIT_VALUE_Q": [5.0, 6.0, 6.5],
        })


