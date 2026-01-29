"""
This script defines functions to clean and prep FDA warning letters
"""

import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa


def list_warning_letter_files(letter_path: str) -> list[str]:
    """
    List all warning letter files in the specified directory.

    Args:
        letter_path (str): Path to the directory containing warning letter files.
    Returns:
        list[str]: List of file paths to warning letter files.
    """
    file_pattern = os.path.join(letter_path, "*announcement.xlsx")
    file_list = glob.glob(file_pattern)
    file_list = [os.path.normpath(p) for p in file_list]
    return file_list


def load_and_prep_warning_letter_file(file_path: str) -> pd.DataFrame:
    """
    Load a warning letter Excel file into a DataFrame.

    Args:
        file_path (str): Path to the warning letter Excel file.
    Returns:
        pd.DataFrame: DataFrame containing the warning letter data.
    """
    df = pd.read_excel(file_path)
    # lowercase columns
    df.columns = df.columns.str.lower()
    # rename columns
    cols_names = ['store_type', 'insp_date', 'issue_date', 'store_name_fda', 'address_fda', 'city', 'state', 'zip_code', 'products']
    df.columns = cols_names
    
    # standardize dtypes
    str_cols = ['store_type', 'store_name_fda', 'address_fda', 'city', 'state', 'zip_code', 'products']
    df[str_cols] = df[str_cols].astype('str')
    
    df['zip_code'] = df['zip_code'].str.replace('\"','')

    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

    # drop online stores
    df = df.loc[~df['store_type'].isin(['Online', 'online'])].copy()

    return df


def build_warning_letter_panel(
        letter_path: str,
        output_file_path: str,
        ) -> pd.DataFrame:
    """
    Build a combined DataFrame of all warning letters in the specified directory.

    Args:
        letter_path (str): Path to the directory containing warning letter files.
        output_path (str): File path for the combined warning letter feather file.
    Returns:
        pd.DataFrame: Concatinated warning letters DataFrame.
    """
    
    output_file_path = os.path.normpath(output_file_path)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    file_list = list_warning_letter_files(letter_path)
    
    df_list = []
    for file_path in file_list:
        df = load_and_prep_warning_letter_file(file_path)
        df_list.append(df)
    
    if not df_list:
        raise ValueError(f"No warning letter files found in directory: {letter_path}")

    combined_df = pd.concat(df_list, ignore_index=True)

    # save panel
    combined_df.to_feather(output_file_path)

    return combined_df