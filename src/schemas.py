# -*- coding: utf-8 -*-
"""
Data contracts for my tables

Role: 
Defines Pandera schemas that describe what my DataFrames should look like:

- StoreSchema: for store-product level monthly data (after we lowercase column names)

- VapeIndexSchema: for the store/date vape price index output

"""

# src/schemas.py

import pandas as pd
import numpy as np
import pandera as pa             # for Field + check decorator
import pandera.pandas as pa_pd   # for DataFrameModel
from pandera.typing import Series


class StoreSchema(pa_pd.DataFrameModel):
    """
    Schema for a single store-level monthly dataset, AFTER column names have
    been lowercased.
    """

    store_id: Series[int]
    calendar_year: Series[int]
    calendar_month: Series[int]

    gtin: Series[str]

    nonscan_category: Series[str] = pa.Field(nullable=True)
    nonscan_subcategory: Series[str] = pa.Field(nullable=True)
    nonscan_detail: Series[str] = pa.Field(nullable=True)

    quantity: Series[float]
    quantity_with_discount: Series[float]
    transaction_count: Series[int]
    transaction_count_with_discount: Series[int]
    total_revenue_amount: Series[float]

    scan_type: Series[str]

    category: Series[str] = pa.Field(nullable=True)
    subcategory: Series[str] = pa.Field(nullable=True)
    manufacturer: Series[str] = pa.Field(nullable=True)
    brand: Series[str] = pa.Field(nullable=True)
    product_type: Series[str] = pa.Field(nullable=True)
    sub_product_type: Series[str] = pa.Field(nullable=True)
    unit_size: Series[str] = pa.Field(nullable=True)
    pack_size: Series[str] = pa.Field(nullable=True)
    product_description: Series[str] = pa.Field(nullable=True)

    q_plus_qwd: Series[float]
    unit_value_q_plus_qwd: Series[float]
    unit_value_q: Series[float]

    class Config:
        strict = False  # allow extra columns if they show up
        coerce = True   # auto-coerce types (e.g. numeric gtin -> str)


class VapeIndexSchema(pa_pd.DataFrameModel):
    """
    Schema for the store-month vape price index output.
    """

    store_id: Series[int]
    date: Series[pd.Timestamp]

    # Allow NaNs but enforce non-negative values
    vape_price_index: Series[float] = pa.Field(ge=0, nullable=True)
    l_vape_price_index: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True

    # ---- custom checks to forbid ±inf ----

    @pa.check("vape_price_index")
    def vape_index_no_infinity(cls, s: pd.Series) -> pd.Series:
        """vape_price_index must not be +inf or -inf."""
        return ~s.isin([np.inf, -np.inf])

    @pa.check("l_vape_price_index")
    def log_vape_index_no_infinity(cls, s: pd.Series) -> pd.Series:
        """l_vape_price_index must not be +inf or -inf."""
        return ~s.isin([np.inf, -np.inf])














