# -*- coding: utf-8 -*-
"""
performs data validation checks (schema checks, missing columns, type checks, etc.)

In Spyder IDE, execute this in console:
!python scripts/validate_data.py
"""

# scripts/validate_data.py

import os
import pandas as pd

from src.validation import validate_store_df, validate_vape_price_index_df

# allow the script to work in spyder AND the terminal
if "__file__" in globals():
    # running as a script
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
else:
    # running interactively (Spyder, Jupyter, IPython console)
    BASE_DIR = os.getcwd()


def main() -> None:
    # sample store-level file (optional)
    sample_store_path = os.path.join(BASE_DIR, "tests", "data", "sample_store.feather")
    if os.path.exists(sample_store_path):
        df_store = pd.read_feather(sample_store_path)
        df_store.columns = df_store.columns.str.lower()
        validate_store_df(df_store)
        print("Sample store-level data validated successfully.")
    else:
        print("No sample_store.feather found, skipping store-level validation.")

    # sample vape index file (optional)
    sample_index_path = os.path.join(BASE_DIR, "tests", "data", "sample_index.feather")
    if os.path.exists(sample_index_path):
        df_index = pd.read_feather(sample_index_path)
        validate_vape_price_index_df(df_index)
        print("Sample vape index data validated successfully.")
    else:
        print("No sample_index.feather found, skipping index validation.")


if __name__ == "__main__":
    main()


