# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 09:06:56 2025

@author: cahase

This script defines the functions that are used to construct store-level monthly
price indexes
"""

import gc
import glob
import os
from timeit import default_timer as timer
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa

from src.validation import validate_store_df, validate_vape_index_df

# ---------------------------------------------------------------------
# Fiscal year helpers
# ---------------------------------------------------------------------


def build_fiscal_year_periods() -> dict:
    """Build mapping of fiscal year -> list of YYYY-MM strings."""
    bfy_2022 = (
        pd.date_range(start="2022-01", end="2022-06", freq="MS")
        .to_period("M")
        .astype("str")
        .tolist()
    )
    bfy_2023 = (
        pd.date_range(start="2022-07", end="2023-06", freq="MS")
        .to_period("M")
        .astype("str")
        .tolist()
    )
    bfy_2024 = (
        pd.date_range(start="2023-07", end="2024-06", freq="MS")
        .to_period("M")
        .astype("str")
        .tolist()
    )
    bfy_2025 = (
        pd.date_range(start="2024-07", end="2025-06", freq="MS")
        .to_period("M")
        .astype("str")
        .tolist()
    )
    # return a dictionary
    return {
        2022: bfy_2022,
        2023: bfy_2023,
        2024: bfy_2024,
        2025: bfy_2025,
    }


def assign_fiscal_year(date_period_series: pd.Series, fy_map: dict) -> pd.Series:
    """
    Given a Period[M] series and a mapping fiscal_year -> list of YYYY-MM strings,
    return a fiscal_year series (int) or NaN.
    """
    dates_str = date_period_series.astype("str")

    fy = np.select(
        [
            dates_str.isin(fy_map[2022]),
            dates_str.isin(fy_map[2023]),
            dates_str.isin(fy_map[2024]),
            dates_str.isin(fy_map[2025]),
        ],
        [2022, 2023, 2024, 2025],
        default=np.nan,
    )
    return pd.Series(fy, index=date_period_series.index)


# ---------------------------------------------------------------------
# Store-level data loading & basic preparation
# ---------------------------------------------------------------------


def list_store_files(store_path: str) -> List[str]:
    """Return a list of all feather files under store_path."""
    pattern = os.path.join(store_path, "*.feather")
    all_files = glob.glob(pattern)
    all_files = [os.path.normpath(p) for p in all_files]
    return all_files


def extract_store_numbers(file_paths: List[str]) -> List[str]:
    """Extract store numbers from file names like '28380.feather'."""
    return [os.path.basename(path).split(".")[0] for path in file_paths]


def load_store_df(path: str) -> pd.DataFrame:
    """Read a store-level feather file and return a DataFrame."""
    return pd.read_feather(path)


def prepare_store_df(df: pd.DataFrame, fy_map: dict) -> pd.DataFrame:
    """
    Basic preparation:
      - lowercase column names
      - create 'date' Period[M] from calendar_year + calendar_month
      - filter to scan_type == 'GTIN'
      - assign fiscal_year
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # date column as Period[M]
    df["date"] = (
        df["calendar_year"].astype("str") + "-" + df["calendar_month"].astype("str")
    )
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M")

    # keep only GTIN scan types
    df = df.loc[df["scan_type"] == "GTIN"].copy()
    if df.empty:
        return df

    # fiscal year assignment
    df["fiscal_year"] = assign_fiscal_year(df["date"], fy_map)

    # Validation check 1: validate structure & dtypes of the store-level data   
    df = validate_store_df(df)
    
    return df


def subset_vaping_products(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where subcategory == 'Vaping Products'."""
    subcat_df = df.loc[df["subcategory"] == "Vaping Products"].copy()
    return subcat_df


# ---------------------------------------------------------------------
# Core index computation for a single store
# ---------------------------------------------------------------------


def compute_unit_value_lags(
        subcat_df: pd.DataFrame,
        value_col: str,
        out_value_col: str = "value",
        out_lag_value_col: str = "lag_value",
        ) -> pd.DataFrame:
    """
    Within each (store_id, gtin) group:
      - sort by date
      - compute month_diff (difference in months between consecutive rows)
      - compute lag of value_col
      - set non-consecutive lags to NaN

    Parameters
    ----------
    value_col : str
        Column to lag (e.g., "unit_value_q" for price index, "quantity" for qty index).
    out_value_col / out_lag_col : str
        Standardized column names used by the rest of the pipeline.
    """
    df = subcat_df.copy()

    if value_col not in df.columns:
        raise KeyError(f"Expected column '{value_col}' not found in DataFrame. Available: {list(df.columns)}")

    df = df.sort_values(by=["store_id", "gtin", "date"])

    # month difference between consecutive rows
    df["month_diff"] = (
        df.groupby(["store_id", "gtin"])["date"]
        .diff()
        .apply(lambda x: x.n if pd.notna(x) else np.nan)
    )

    # standardize the raw value column name
    df[out_value_col] = df[value_col]

    # lagged unit_value_q
    df[out_lag_value_col] = df.groupby(["store_id", "gtin"])[out_value_col].shift(1)

    # keep only lags for consecutive months
    df[out_lag_value_col] = np.where(df["month_diff"] == 1, df[out_lag_value_col], np.nan)

    df = df.fillna(value=np.nan)

    return df


def compute_revenue_weights(subcat_df: pd.DataFrame, period_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute category_annual_revenue, type_annual_revenue, and product_annual_revenue
    for subcategory 'Vaping Products'.
    - period_col = "fiscal_year" or "calendar_year"
    """
    # (sub)category annual revenue
    category_revenue = (
        subcat_df.groupby(["store_id", "subcategory", period_col])
        .agg(category_annual_revenue=("total_revenue_amount", "sum"))
        .reset_index()
    )

    # product type annual revenue
    type_revenue = (
        subcat_df.groupby(
            ["store_id", "subcategory", "product_type", period_col], dropna=False
        )
        .agg(type_annual_revenue=("total_revenue_amount", "sum"))
        .reset_index()
    )

    # product annual revenue
    product_revenue = (
        subcat_df.groupby(
            ["store_id", "subcategory", "product_type", "gtin", period_col],
            dropna=False,
        )
        .agg(product_annual_revenue=("total_revenue_amount", "sum"))
        .reset_index()
    )

    return category_revenue, type_revenue, product_revenue


def compute_vape_price_index_for_store(
        subcat_df: pd.DataFrame, 
        weight_basis: str = "fiscal",
        index_kind: str = "price",
        ) -> pd.DataFrame: 
    """
    Compute store-level monthly index and its log for a single store's Vaping Products.
    Returns a DataFrame with columns: store_id, date, AND (vape_price_index, l_vape_price_index) 
    OR (vape_qty_index, l_vape_qty_index).subcategory is dropped at the end.

    weight_basis: 'fiscal' or 'calendar'
    index_kind: 'price' (unit_value_q) or 'qty' (quantity)
    """
    if weight_basis not in {"fiscal", "calendar"}:
        raise ValueError("weight_basis must be 'fiscal' or 'calendar'")
    
    if index_kind not in {"price", "qty"}:
        raise ValueError("index_kind must be 'price' or 'qty'")
    
    period_col = "fiscal_year" if weight_basis == "fiscal" else "calendar_year"
    value_col = "unit_value_q" if index_kind == "price" else "quantity"

    # compute lagged unit values & month_diff
    subcat_df = compute_unit_value_lags(subcat_df, value_col=value_col)

    # revenue weights
    category_revenue, type_revenue, product_revenue = compute_revenue_weights(subcat_df, period_col)

    # stage 1 weight: product share of type revenue
    stage_1_weight = pd.merge(
        product_revenue,
        type_revenue,
        how="left",
        on=["store_id", "subcategory", "product_type", period_col],
    )
    stage_1_weight["stage_1_weight"] = (
        stage_1_weight["product_annual_revenue"] / stage_1_weight["type_annual_revenue"]
    )

    # merge stage 1 weight onto transactional data
    stage_1_df = pd.merge(
        subcat_df,
        stage_1_weight,
        how="left",
        on=["store_id", "subcategory", "product_type", "gtin", period_col],
    )

    # stage 1: unit value index
    stage_1_df["unit_value_index"] = (
        stage_1_df["value"] / stage_1_df["lag_value"]
        ) ** stage_1_df["stage_1_weight"]

    # replace +/-inf caused by division-by-zero, or non-positive values -> nan
    bad_level = (
        ~np.isfinite(stage_1_df["unit_value_index"])
        | (stage_1_df["unit_value_index"] <= 0)
    )
    if bad_level.any():
        stage_1_df.loc[bad_level, "unit_value_index"] = np.nan

    # # replace +/-inf caused by division-by-zero or extreme ratios
    # stage_1_df["unit_value_index"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # aggregate to product_type index
    type_index = (
        stage_1_df
        .groupby(["store_id", "subcategory", "product_type", "date"], dropna=False)
        .agg(type_index=("unit_value_index", lambda s: s.prod(min_count=1)))
        .reset_index()
    )

    # stage 2 weight: type share of category
    stage_2_weight = pd.merge(
        type_revenue,
        category_revenue,
        how="left",
        on=["store_id", "subcategory", period_col],
    )
    stage_2_weight["stage_2_weight"] = (
        stage_2_weight["type_annual_revenue"]
        / stage_2_weight["category_annual_revenue"]
    )

    # fiscal year lookup by date
    fisc_year_date_df = stage_1_df[["date", period_col]].drop_duplicates()

    # attach fiscal_year to each date in type_index
    type_index = pd.merge(
        type_index,
        fisc_year_date_df,
        how="left",
        on=["date"],
    )

    # merge stage 2 weights
    stage_2_df = pd.merge(
        type_index,
        stage_2_weight,
        how="left",
        on=["store_id", "subcategory", "product_type", period_col],
    )

    # replace +/-inf caused by division-by-zero, or non-positive values -> nan
    bad_level = (
        ~np.isfinite(stage_2_df["type_index"])
        | (stage_2_df["type_index"] <= 0)
    )
    if bad_level.any():
        stage_2_df.loc[bad_level, "type_index"] = np.nan
    
    # stage 2: weighted type index
    stage_2_df["weighted_type_index"] = stage_2_df["type_index"] ** stage_2_df["stage_2_weight"]

    # Choose output column names based on index_kind
    index_col = "vape_price_index" if index_kind == "price" else "vape_qty_index"
    log_col = f"l_{index_col}"

    # aggregate to store-level vape price index
    store_index = (
        stage_2_df
        .groupby(["store_id", "subcategory", "date"])
        .agg(**{index_col: ("weighted_type_index", lambda s: s.prod(min_count=1))})
        .reset_index()
        .drop(columns="subcategory")
    )

    # clean the level index: any non-finite or non-positive values -> nan
    bad_level = (
        ~np.isfinite(store_index[index_col])
        | (store_index[index_col] <= 0)
    )
    if bad_level.any():
        store_index.loc[bad_level, index_col] = np.nan

    # log index, suppress divide-by-zero warnings (since we expect some and fix them below)
    with np.errstate(divide="ignore", invalid="ignore"):
        store_index[log_col] = np.log(
            store_index[index_col]
        )

    # replace any remaining +/-inf in the log with nan
    store_index[log_col] = store_index[log_col].replace(
        [np.inf, -np.inf], np.nan
    )
    
    # store_index["l_vape_price_index"] = np.log(store_index["vape_price_index"])

    return store_index


# ---------------------------------------------------------------------
# Multi-store processing and panel building
# ---------------------------------------------------------------------


def process_store_file(
        store: str, 
        store_path: str, 
        outpath: str, 
        fy_map: dict, 
        weight_basis: str = "fiscal",
        index_kind: str = "price",
        ) -> bool:
    """
    Process a single store:
      - read in feather file
      - prepare dataframe
      - subset to vaping
      - construct index
      - save per-store index feather

    Returns True if processed, False if skipped (empty GTIN or empty vaping).
    """
    input_filename = os.path.join(store_path, f"{store}.feather")
    input_filename = os.path.normpath(input_filename)

    df = load_store_df(input_filename)
    df = prepare_store_df(df, fy_map)

    if df.empty:
        print(f"Skipping {store} — empty DataFrame after GTIN filter")
        return False

    subcat_df = subset_vaping_products(df)
    if subcat_df.empty:
        print(f"Skipping {store} — empty Vaping Products subset")
        return False

    store_index = compute_vape_price_index_for_store(
        subcat_df, 
        weight_basis=weight_basis, 
        index_kind=index_kind
        )

    # Validation check 2: validate the per-store vape index (no +/-inf, correct dtypes, etc.)
    store_index = validate_vape_index_df(store_index)
    
    output_filename = os.path.join(outpath, f"{store}.feather")
    output_filename = os.path.normpath(output_filename)
    pa.feather.write_feather(store_index, output_filename)

    return True


def process_all_stores(
        store_path: str, 
        outpath: str, 
        weight_basis: str = "fiscal",
        index_kind: str = "price",
        limit: int | None = None
        ) -> None:
    """
    Loop over all store files, compute indexes and save per-store results.
    Added optional limit of store numbers for dry runs
    """
    os.makedirs(outpath, exist_ok=True)

    all_files = list_store_files(store_path)
    store_numbers = extract_store_numbers(all_files)
    
    # in case of dry run
    if limit is not None:
        store_numbers = store_numbers[:limit]

    # create mapping (dict) relating fiscal years to dates
    fy_map = build_fiscal_year_periods()

    total_iterations = len(store_numbers)
    iteration = 0

    start = timer()
    for store in store_numbers:
        processed = process_store_file(
            store, 
            store_path, 
            outpath, 
            fy_map, 
            weight_basis=weight_basis, 
            index_kind=index_kind
            )
        iteration += 1
        status = "Processed" if processed else "Skipped"
        print(f"Iteration {iteration}/{total_iterations}: {status} store {store}")

        # optional clean-up for big loops
        gc.collect()

    end = timer()
    print(f"Finished processing all stores in {end - start:.2f} seconds.")


def build_panel_index(source_dir: str, output_path: str) -> pd.DataFrame:
    """
    Read all per-store index feather files, concatenate into a panel, drop duplicates,
    convert store_id to string (as in your original), and save to output_path.
    """
    pattern = os.path.join(source_dir, "*.feather")
    all_files = glob.glob(pattern)
    all_files = [os.path.normpath(p) for p in all_files]

    if not all_files:
        raise RuntimeError(
            f"No store-level index files found in: {source_dir}. "
            "Upstream processing produced zero outputs."
            )

    total_iterations = len(all_files)
    print(f"Building panel from {total_iterations} store-level index files.")

    li = []
    for i, file in enumerate(all_files, start=1):
        df = pd.read_feather(file)
        li.append(df)
        print(f"Read {i}/{total_iterations} files")

    indexes = pd.concat(li, ignore_index=True)

    indexes = indexes.drop_duplicates(subset=["store_id", "date"])
    
    # Validation check 3: validate final index panel (no ±inf, correct dtypes, etc.)
    indexes = validate_vape_index_df(indexes)

    # save final panel
    indexes.to_feather(output_path)
    print(f"Saved panel index to {output_path}")

    return indexes

