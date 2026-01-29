# -*- coding: utf-8 -*-
"""
Reusable validation helpers to validate data. These are small 
functions I can call in:
    - tests / CI (for sample data)
    - my pipeline (for real data)
"""

import pandas as pd
from src.schemas import StoreSchema, VapePriceIndexSchema, VapeQtyIndexSchema


def validate_store_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({None: pd.NA})
    # DataFrameModel.validate works the same way
    return StoreSchema.validate(df)


def validate_vape_price_index_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({None: pd.NA})

    if isinstance(df["date"].dtype, pd.PeriodDtype):
        df["date"] = df["date"].dt.to_timestamp()

    validated = VapePriceIndexSchema.validate(df)

    if validated.duplicated(subset=["store_id", "date"]).any():
        raise ValueError("Duplicate (store_id, date) combinations in vape price index DataFrame.")

    return validated


def validate_vape_qty_index_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({None: pd.NA})

    if isinstance(df["date"].dtype, pd.PeriodDtype):
        df["date"] = df["date"].dt.to_timestamp()

    validated = VapeQtyIndexSchema.validate(df)

    if validated.duplicated(subset=["store_id", "date"]).any():
        raise ValueError("Duplicate (store_id, date) combinations in vape quantity index DataFrame.")

    return validated































